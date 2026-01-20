#include "sentry_communicator/can_bus.hpp"
#define MAX_SPEED 10.f
#define MIN_SPEED -10.f
#define MAX_ANGLE 180.f
#define MIN_ANGLE -180.f

namespace sentry_communicator
{

  CanBus::CanBus(const std::string &bus_name, int thread_priority) : Node("can_bus_node"), bus_name_(bus_name)
  {
    this->declare_parameter("debug", false);
    this->get_parameter("debug", debug_);
    referee_pub_ = this->create_publisher<robot_msg::msg::RefereeInfoMsg>("/referee_info", 10);
    team_hp_pub_ = this->create_publisher<robot_msg::msg::RobotHpMsg>("/Team_robot_HP", 10);
    enemy_hp_pub_ = this->create_publisher<robot_msg::msg::RobotHpMsg>("/Enemy_robot_HP", 10);
    area_status_pub_ = this->create_publisher<robot_msg::msg::AreaStatusMsg>("/centrum_status", 10);
    yaw_speed_ekf_ = std::make_shared<tools::ExtendedKalmanFilter>(yaw_speed_x0, yaw_speed_P0);
    yaw_speed_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/yaw_speed", 10);
    chassis_info_pub_ = this->create_publisher<robot_msg::msg::ChassisMsg>("/chassis_info", 10);
    global_relocalization_pub_ = this->create_publisher<robot_msg::msg::GlobalRelocalizationMsg>(
        "/global_relocalization_param", 10);
    path_subscription_ = this->create_subscription<nav_msgs::msg::Path>(
        "/plan", 10, std::bind(&CanBus::planCallback, this, std::placeholders::_1));
    cmd_rotate_mode_sub_ = this->create_subscription<robot_msg::msg::RotateModeMsg>(
        "/sentry/rotate_mode", 10, std::bind(&CanBus::cmdRotateModeCallback, this, std::placeholders::_1));
    lio_state_sub_ = this->create_subscription<robot_msg::msg::LioStateMsg>(
        "/lio_state", 10, std::bind(&CanBus::lioStateCallback, this, std::placeholders::_1));
    while (!socket_can_.open(bus_name, std::bind(&CanBus::frameCallback, this, std::placeholders::_1), thread_priority) &&
           rclcpp::ok())
    {
      RCLCPP_INFO(this->get_logger(), "[CAN_BUS] : Trying to connect to %s...", bus_name.c_str());
      rclcpp::sleep_for(std::chrono::milliseconds(500));
    }
    RCLCPP_INFO(this->get_logger(), "[CAN_BUS] : Successfully connected to %s.", bus_name.c_str());

    chassis_buffer_.writeFromNonRT(geometry_msgs::msg::Twist());
    gimbal_buffer_.writeFromNonRT(robot_msg::msg::GimbalControlMsg());

    cmd_chassis_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/sentry/cmd_vel", 10, std::bind(&CanBus::cmdChassisCallback, this, std::placeholders::_1));
    cmd_gimbal_sub_ = this->create_subscription<robot_msg::msg::GimbalControlMsg>(
        "/sentry/cmd_gimbal", 10, std::bind(&CanBus::cmdGimbalCallback, this, std::placeholders::_1));
    gimbal_stop_sub_ = this->create_subscription<robot_msg::msg::GimbalStopMsg>(
        "/sentry/gimbal_stop", 10, std::bind(&CanBus::gimbalStopCallback, this, std::placeholders::_1));
    chassis_frame_.can_id = 0x111;
    chassis_frame_.can_dlc = 8;

    gimbal_frame_.can_id = 0x112;
    gimbal_frame_.can_dlc = 8;

    gimbal_stop_frame_.can_id = 0x109;
    gimbal_stop_frame_.can_dlc = 8;

    path_frame_.can_id = 0x114;
    path_frame_.can_dlc = 8;
    timer_ = this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&CanBus::write, this));
    path_timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&CanBus::path_write, this));
  }

  void CanBus::path_write()
  {
    if (path_get_flag_.load() == false)
    {
      path_write_flag_.store(false);
      return;
    }
    path_write_flag_.store(true);
  }
  void CanBus::write()
  {
    if (path_write_flag_.load() == true)
    {
      path_mutex_.lock();
      std::fill(std::begin(path_frame_.data), std::end(path_frame_.data), 0);
      if (path_update_ == true)
      {
        path_update_ = false;
        frame_index_ = 1;
      }
      path_frame_.data[0] = static_cast<uint8_t>(frame_index_);
      if (frame_index_ == 1)
      {
        path_frame_.data[1] = static_cast<uint8_t>(3);
        path_frame_.data[2] = static_cast<uint8_t>(path_start_x_ >> 8);
        path_frame_.data[3] = static_cast<uint8_t>(path_start_x_);
        path_frame_.data[4] = static_cast<uint8_t>(path_start_y_ >> 8);
        path_frame_.data[5] = static_cast<uint8_t>(path_start_y_);
      }
      else if (frame_index_ <= 8) // 写入x
      {
        for (int i = 1; i < 8; i++)
        {
          if ((frame_index_ - 2) * 7 + i <= path_size_)
          {
            // path_frame_.data[i] = static_cast<uint8_t>(path_deltas_[((frame_index_ - 1) * 7 + i) - 1].first);
            path_frame_.data[i] = static_cast<uint8_t>(1);
          }
          // std::cout<<"i :"<<(frame_index_ - 2) * 7 + i<<std::endl;
        }
      }
      else if (frame_index_ > 8 && frame_index_ <= 15) // 写入y
      {
        for (int i = 1; i < 8; i++)
        {
          if ((frame_index_ - 9) * 7 + i <= path_size_)
          {
            // path_frame_.data[i] = static_cast<uint8_t>(path_deltas_[((frame_index_ - 1) * 7 + i) - 1].second);
            path_frame_.data[i] = static_cast<uint8_t>(1);
          }
        }
      }
      socket_can_.write(&path_frame_);
      frame_index_++;
      if (frame_index_ > 15) // 传输完成15帧，从零开始计数
      {
        frame_index_ = 1;
        path_write_flag_.store(false);
      }
      path_mutex_.unlock();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const robot_msg::msg::RotateModeMsg *rotate_mode = rotate_mode_buffer_.readFromRT();

    std::fill(std::begin(chassis_frame_.data), std::end(chassis_frame_.data), 0);

    const geometry_msgs::msg::Twist *cmd_vel = chassis_buffer_.readFromRT();

    if (cmd_vel)
    {
      float vel_x_origin = cmd_vel->linear.x;
      float vel_y_origin = cmd_vel->linear.y;
      float vel_z_origin = rotate_mode->roate_velocity;
      bool lio_running = lio_running_.load();
      if (!lio_running)
      {
        vel_x_origin = 0.0;
        vel_y_origin = 0.0;
        vel_z_origin = 10.0;
      }
      uint16_t vel_x = float2uint(vel_x_origin, MIN_SPEED, MAX_SPEED, 12);
      uint16_t vel_y = float2uint(vel_y_origin, MIN_SPEED, MAX_SPEED, 12);
      uint16_t vel_z = float2uint(vel_z_origin, MIN_SPEED, MAX_SPEED, 12);

      chassis_frame_.data[0] = static_cast<uint8_t>(vel_x >> 4u);
      chassis_frame_.data[1] = static_cast<uint8_t>((vel_x & 0xF) << 4u | vel_y >> 8u);
      chassis_frame_.data[2] = static_cast<uint8_t>(vel_y);
      chassis_frame_.data[3] = static_cast<uint8_t>(vel_z >> 4u);
      chassis_frame_.data[4] = static_cast<uint8_t>((vel_z & 0xF) << 4u | 0xF);
      if (rotate_mode->mode == 1)
      {
        // 启用底盘跟随
        chassis_frame_.data[5] = 0xFF;
      }
      else
      {
        // 禁用底盘跟随
        chassis_frame_.data[5] = 0x00;
      }
      socket_can_.write(&chassis_frame_);
    }

    std::fill(std::begin(gimbal_frame_.data), std::end(gimbal_frame_.data), 0);
    const robot_msg::msg::GimbalControlMsg *gimbal_cmd = gimbal_buffer_.readFromRT();
    uint16_t yaw_lower_limit = float2uint(gimbal_cmd->yaw_lower_limit, MIN_ANGLE, MAX_ANGLE, 16);
    uint16_t yaw_upper_limit = float2uint(gimbal_cmd->yaw_upper_limit, MIN_ANGLE, MAX_ANGLE, 16);
    uint16_t pitch_lower_limit = float2uint(gimbal_cmd->pitch_lower_limit, MIN_ANGLE, MAX_ANGLE, 16);
    uint16_t pitch_upper_limit = float2uint(gimbal_cmd->pitch_upper_limit, MIN_ANGLE, MAX_ANGLE, 16);
    gimbal_frame_.data[0] = static_cast<uint8_t>(yaw_lower_limit >> 8);
    gimbal_frame_.data[1] = static_cast<uint8_t>(yaw_lower_limit);
    gimbal_frame_.data[2] = static_cast<uint8_t>(yaw_upper_limit >> 8);
    gimbal_frame_.data[3] = static_cast<uint8_t>(yaw_upper_limit);
    gimbal_frame_.data[4] = static_cast<uint8_t>(pitch_lower_limit >> 8);
    gimbal_frame_.data[5] = static_cast<uint8_t>(pitch_lower_limit);
    gimbal_frame_.data[6] = static_cast<uint8_t>(pitch_upper_limit >> 8);
    gimbal_frame_.data[7] = static_cast<uint8_t>(pitch_upper_limit);
    socket_can_.write(&gimbal_frame_);

    std::fill(std::begin(gimbal_stop_frame_.data), std::end(gimbal_stop_frame_.data), 0);
    bool lio_running = lio_running_.load();
    if (gimbal_stop_flag_.load() || !lio_running)
    {
      gimbal_stop_frame_.data[0] = 0xFF; // stop gimabl
    }
    else
    {
      gimbal_stop_frame_.data[0] = 0x00;
    }
    socket_can_.write(&gimbal_stop_frame_);
  }
  void CanBus::cmdRotateModeCallback(const robot_msg::msg::RotateModeMsg::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    rotate_mode_buffer_.writeFromNonRT(*msg);
  }
  void CanBus::gimbalStopCallback(const robot_msg::msg::GimbalStopMsg::SharedPtr msg)
  {
    if (msg->gimbal_stop == 1)
      gimbal_stop_flag_.store(true);
    else
      gimbal_stop_flag_.store(false);
  }
  void CanBus::lioStateCallback(const robot_msg::msg::LioStateMsg::SharedPtr msg)
  {
    // 超时后不再接收数据
    if (msg->state == 0 || over_wait_time_ == true)
    {
      lio_stop_ = false;
      lio_running_.store(true);
    }
    else if (lio_stop_ == false)
    {
      lio_stop_ = true;
      lio_stop_time_ = std::chrono::high_resolution_clock::now();
      lio_running_.store(false);
    }
    else
    {
      auto now = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lio_stop_time_);
      if (duration.count() > 30000) // 30s超时
      {
        over_wait_time_ = true;
        lio_stop_ = false;
        lio_running_.store(true);
      }
    }
  }

  // void CanBus::planCallback(const nav_msgs::msg::Path::SharedPtr msg)
  // {
  //   path_get_flag_.store(true);
  //   path_mutex_.lock();
  //   size_t N = msg->poses.size();
  //   if (N < 5)
  //   {
  //     path_mutex_.unlock();
  //     RCLCPP_WARN(rclcpp::get_logger("PlanConverter"), "plan too short, N = %zu", N);
  //     return;
  //   }
  //   path_update_ = true;
  //   // 1. 确定采样后的点数 M
  //   size_t M = std::min<size_t>(N, 10);
  //   path_size_ = static_cast<int>(M);
  //   // 2. 等距下采样到 M 个点
  //   std::vector<geometry_msgs::msg::PoseStamped> sampled;
  //   sampled.reserve(M);
  //   for (size_t i = 0; i < M; ++i)
  //   {
  //     // 映射 i in [0, M-1] 到 idx in [0, N-1]
  //     size_t idx = (N == 1 ? 0 : (i * (N - 1) / (M - 1)));
  //     sampled.push_back(msg->poses[idx]);
  //   }
  //   // 3. 起点坐标转换
  //   double x0_m = sampled[0].pose.position.x;
  //   double y0_m = sampled[0].pose.position.y;
  //   path_start_x_ = static_cast<int>(std::round(x0_m * 10.0));
  //   path_start_y_ = static_cast<int>(std::round(y0_m * 10.0));
  //   // 4. 计算增量数组
  //   path_deltas_.clear();
  //   path_deltas_.reserve(49);
  //   for (size_t i = 1; i < M; ++i)
  //   {
  //     double dx_m = sampled[i].pose.position.x - sampled[i - 1].pose.position.x;
  //     double dy_m = sampled[i].pose.position.y - sampled[i - 1].pose.position.y;
  //     int dx = static_cast<int>(std::round(dx_m * 10.0));
  //     int dy = static_cast<int>(std::round(dy_m * 10.0));
  //     path_deltas_.emplace_back(dx, dy);
  //   }
  //   // 6. 输出或进一步处理
  //   std::ostringstream oss;
  //   oss << "起点(dm): (" << static_cast<int>(path_start_x_) << ", " << static_cast<int>(path_start_y_) << "); ";
  //   oss << "点数: " << path_size_ << "; 增量数组: [";
  //   for (size_t i = 0; i < path_deltas_.size(); ++i)
  //   {
  //     oss << "("
  //         << (path_deltas_[i].first) << ","
  //         << (path_deltas_[i].second) << ")";
  //     if (i + 1 < path_deltas_.size())
  //       oss << ", ";
  //   }
  //   oss << "]";
  //   RCLCPP_INFO(rclcpp::get_logger("PlanConverter"), "%s", oss.str().c_str());
  //   path_mutex_.unlock();
  // }
  void CanBus::cmdChassisCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    chassis_buffer_.writeFromNonRT(*msg);
  }
  void CanBus::cmdGimbalCallback(const robot_msg::msg::GimbalControlMsg::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    gimbal_buffer_.writeFromNonRT(*msg);
  }
  void CanBus::frameCallback(const can_frame &frame)
  {
    /* --------------------------------- 轮速计、yaw角度 --------------------------------- */
    if (frame.can_id == 0x340)
    {
      chassis_msg_.stamp = this->now();
      int16_t chassis_vx = (int16_t)(frame.data[0] << 8 | frame.data[1]);
      int16_t chassis_vy = (int16_t)(frame.data[2] << 8 | frame.data[3]);
      int16_t chassis_vw = (int16_t)(frame.data[4] << 8 | frame.data[5]);
      int16_t chassis_yaw = (int16_t)(frame.data[6] << 8 | frame.data[7]);
      chassis_msg_.vx = uint2float(chassis_vx, -5.0f, 5.0f, 16);
      chassis_msg_.vy = uint2float(chassis_vy, -5.0f, 5.0f, 16);
      chassis_msg_.wz = uint2float(chassis_vw, -15.0f, 15.0f, 16);
      chassis_msg_.yaw = uint2float(chassis_yaw, -M_PI, M_PI, 16);

      double current_time = rclcpp::Time(chassis_msg_.stamp).seconds();
      if (!first_yaw_speed_)
      {
        double dt = current_time - last_time_;
        if (dt > 1e-6)
        {
          double yaw_diff = chassis_msg_.yaw - last_yaw_;
          // 处理角度跳变
          if (yaw_diff > M_PI)
            yaw_diff -= 2.0 * M_PI;
          else if (yaw_diff < -M_PI)
            yaw_diff += 2.0 * M_PI;
          chassis_msg_.yaw_speed = yaw_diff / dt;
        }
      }
      else
      {
        first_yaw_speed_ = false;
        chassis_msg_.yaw_speed = 0.0;
      }
      last_time_ = current_time;
      last_yaw_ = chassis_msg_.yaw;

      chassis_info_pub_->publish(chassis_msg_);
    }
  }

} // namespace sentry_communicator
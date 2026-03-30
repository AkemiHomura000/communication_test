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
    team_hp_pub_ = this->create_publisher<robot_msg::msg::RobotHpMsg>("/team_robot_hp", 10);
    posture_feedback_pub_ = this->create_publisher<robot_msg::msg::PostureFeedbackMsg>("/sentry/posture_feedback", 10);
    chassis_info_pub_ = this->create_publisher<robot_msg::msg::ChassisMsg>("/chassis_info", 10);
    posture_control_sub_ = this->create_subscription<robot_msg::msg::PostureControlMsg>(
        "/sentry/posture_control", 10, std::bind(&CanBus::postureControlCallback, this, std::placeholders::_1));
    while (!socket_can_.open(bus_name, std::bind(&CanBus::frameCallback, this, std::placeholders::_1), thread_priority) &&
           rclcpp::ok())
    {
      RCLCPP_INFO(this->get_logger(), "[CAN_BUS] : Trying to connect to %s...", bus_name.c_str());
      rclcpp::sleep_for(std::chrono::milliseconds(500));
    }
    RCLCPP_INFO(this->get_logger(), "[CAN_BUS] : Successfully connected to %s.", bus_name.c_str());

    chassis_buffer_.writeFromNonRT(geometry_msgs::msg::Twist());
    posture_buffer_.writeFromNonRT(robot_msg::msg::PostureControlMsg());
    chassis_mode_buffer_.writeFromNonRT(robot_msg::msg::ChassisModeMsg());
    cmd_chassis_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/sentry/cmd_vel", 10, std::bind(&CanBus::cmdChassisCallback, this, std::placeholders::_1));
    chassis_mode_sub_ = this->create_subscription<robot_msg::msg::ChassisModeMsg>(
        "/chassis/mode", 10, std::bind(&CanBus::chassisModeCallback, this, std::placeholders::_1));
    chassis_frame_.can_id = 0x111;
    chassis_frame_.can_dlc = 8;

    posture_frame_.can_id = 0x115;
    posture_frame_.can_dlc = 8;

    timer_ = this->create_wall_timer(std::chrono::milliseconds(5), std::bind(&CanBus::write, this));
  }

  void CanBus::write()
  {
    std::lock_guard<std::mutex> lock(mutex_);

    std::fill(std::begin(chassis_frame_.data), std::end(chassis_frame_.data), 0);

    const geometry_msgs::msg::Twist *cmd_vel = chassis_buffer_.readFromRT();
    const robot_msg::msg::ChassisModeMsg *chassis_mode = chassis_mode_buffer_.readFromRT();

    if (cmd_vel)
    {
      float vel_x_origin = cmd_vel->linear.x;
      float vel_y_origin = cmd_vel->linear.y;
      float vel_z_origin = 0.0f;
      if (chassis_mode->mode == 0)
      {
        vel_z_origin = chassis_mode->rotate_velocity;
      }

      bool enable_motion = (referee_msg_.game_progress == 4);
      if (!enable_motion&& !debug_)
      {
        vel_x_origin = 0.0f;
        vel_y_origin = 0.0f;
        RCLCPP_INFO(this->get_logger(),"set velocity =0");
      }
      uint16_t vel_x = float2uint(vel_x_origin, MIN_SPEED, MAX_SPEED, 12);
      uint16_t vel_y = float2uint(vel_y_origin, MIN_SPEED, MAX_SPEED, 12);
      uint16_t vel_z = float2uint(vel_z_origin, MIN_SPEED, MAX_SPEED, 12);

      chassis_frame_.data[0] = static_cast<uint8_t>(vel_x >> 4u);
      chassis_frame_.data[1] = static_cast<uint8_t>((vel_x & 0xF) << 4u | vel_y >> 8u);
      chassis_frame_.data[2] = static_cast<uint8_t>(vel_y);
      chassis_frame_.data[3] = static_cast<uint8_t>(vel_z >> 4u);
      chassis_frame_.data[4] = static_cast<uint8_t>((vel_z & 0xF) << 4u | 0xF);
      if (chassis_mode->mode == 1)
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

    // 姿态控制帧
    std::fill(std::begin(posture_frame_.data), std::end(posture_frame_.data), 0);
    const robot_msg::msg::PostureControlMsg *posture_cmd = posture_buffer_.readFromRT();
    if (posture_cmd)
    {
      posture_frame_.data[0] = posture_cmd->posture_type;  // 始终发送当前姿态类型
    }
    socket_can_.write(&posture_frame_);
  }

  void CanBus::cmdChassisCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    chassis_buffer_.writeFromNonRT(*msg);
  }
  void CanBus::postureControlCallback(const robot_msg::msg::PostureControlMsg::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    posture_buffer_.writeFromNonRT(*msg);
  }
  void CanBus::chassisModeCallback(const robot_msg::msg::ChassisModeMsg::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    chassis_mode_buffer_.writeFromNonRT(*msg);
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
    /* --------------------------------- 己方血量、 referee信息 -------------------------------- */
    if (frame.can_id == 0x101)
    {
      referee_msg_.game_progress = frame.data[0];
      uint16_t stage_remain_time = static_cast<uint16_t>((frame.data[1] << 8) | frame.data[2]);
      referee_msg_.stage_remain_time = stage_remain_time;

      uint16_t current_hp = static_cast<uint16_t>((frame.data[3] << 8) | frame.data[4]);
      team_hp_msg_.sentry_hp = current_hp;

      uint16_t projectile_allowance = static_cast<uint16_t>((frame.data[5] << 8) | frame.data[6]);
      referee_msg_.projectile_allowance = projectile_allowance;

      bool topple_mode = (frame.data[7] == 0x01);
      referee_msg_.key = topple_mode ? 1 : 0;

      if (!debug_)
      {
        referee_pub_->publish(referee_msg_);
        team_hp_pub_->publish(team_hp_msg_);
      }
    }


    /* --------------------------------- 姿态反馈 -------------------------------- */
    if (frame.can_id == 0x116)
    {
      posture_feedback_mutex_.lock();
      posture_feedback_.current_posture = frame.data[0];

      uint16_t time_since_switch_raw = (frame.data[1] << 8) | frame.data[2];
      uint16_t posture_duration_raw = (frame.data[3] << 8) | frame.data[4];
      
      posture_feedback_.time_since_last_switch = static_cast<float>(time_since_switch_raw) * 0.1f;
      posture_feedback_.current_posture_duration = static_cast<float>(posture_duration_raw) * 0.1f;
      posture_feedback_.can_switch = (frame.data[5] == 0xFF);
      posture_feedback_.be_attacked = (static_cast<int8_t>(frame.data[6]) == static_cast<int8_t>(00000001));
      
      if (!debug_)
        posture_feedback_pub_->publish(posture_feedback_);
      posture_feedback_mutex_.unlock();
    }
    /* --------------------------------- 热量信息 -------------------------------- */
    if (frame.can_id == 0xAB)
    {
      posture_feedback_mutex_.lock();
      // 解析17mm枪管热量（16位数据,高字节在前）
      // 下位机在 data[0] 和 data[1] 发送：data[0] = (heat >> 8), data[1] = heat
      uint16_t barrel_heat = (frame.data[0] << 8) | frame.data[1];
      posture_feedback_.shooter_17mm_1_barrel_heat = barrel_heat;
      
      if (!debug_)
        posture_feedback_pub_->publish(posture_feedback_);
      posture_feedback_mutex_.unlock();
    }
  }

} // namespace sentry_communicator
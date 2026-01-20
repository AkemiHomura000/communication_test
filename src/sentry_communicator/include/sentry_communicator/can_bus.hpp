#pragma once

#include <string>
#include <mutex>
#include <algorithm> // For std::fill

// ROS 2
#include "geometry_msgs/msg/twist.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <realtime_tools/realtime_buffer.h>

#include <queue>
#include <atomic>
#include "ekf.hpp"
// Superpower_hardware
#include <sentry_communicator/socketcan.h>
#include <sensor_msgs/msg/imu.hpp>
#include "robot_msg/msg/gimbal_control_msg.hpp"
#include "robot_msg/msg/referee_info_msg.hpp"
#include "robot_msg/msg/robot_hp_msg.hpp"
#include "robot_msg/msg/rotate_mode_msg.hpp"
#include "robot_msg/msg/area_status_msg.hpp"
#include "robot_msg/msg/gimbal_stop_msg.hpp"
#include "robot_msg/msg/lio_state_msg.hpp"
#include "robot_msg/msg/global_relocalization_msg.hpp"
#include "robot_msg/msg/chassis_msg.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <cmath>
#include <vector>
#include <cstdint>
#include <chrono>
#include "nav_msgs/msg/path.hpp"
#define MAX_SPEED 10.f
#define MIN_SPEED -10.f
#define MAX_ANGLE 180.f
#define MIN_ANGLE -180.f

namespace sentry_communicator
{
  struct CanFrameStamp
  {
    can_frame frame;
    rclcpp::Time stamp;
  };

  class CanBus : public rclcpp::Node
  {
  public:
    CanBus(const std::string &bus_name, int thread_priority);
    void write();
    void path_write();

  private:
    bool debug_;
    void frameCallback(const can_frame &frame);
    void cmdChassisCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void cmdGimbalCallback(const robot_msg::msg::GimbalControlMsg::SharedPtr msg);
    void cmdRotateModeCallback(const robot_msg::msg::RotateModeMsg::SharedPtr msg);
    void gimbalStopCallback(const robot_msg::msg::GimbalStopMsg::SharedPtr msg);
    void lioStateCallback(const robot_msg::msg::LioStateMsg::SharedPtr msg);
    void planCallback(const nav_msgs::msg::Path::SharedPtr msg);
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_chassis_sub_;
    rclcpp::Subscription<robot_msg::msg::GimbalControlMsg>::SharedPtr cmd_gimbal_sub_;
    rclcpp::Subscription<robot_msg::msg::RotateModeMsg>::SharedPtr cmd_rotate_mode_sub_;
    rclcpp::Subscription<robot_msg::msg::GimbalStopMsg>::SharedPtr gimbal_stop_sub_;
    rclcpp::Subscription<robot_msg::msg::LioStateMsg>::SharedPtr lio_state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr yaw_speed_pub_;
    rclcpp::Publisher<robot_msg::msg::RefereeInfoMsg>::SharedPtr referee_pub_;
    rclcpp::Publisher<robot_msg::msg::RobotHpMsg>::SharedPtr team_hp_pub_;
    rclcpp::Publisher<robot_msg::msg::RobotHpMsg>::SharedPtr enemy_hp_pub_;
    rclcpp::Publisher<robot_msg::msg::AreaStatusMsg>::SharedPtr area_status_pub_;
    rclcpp::Publisher<robot_msg::msg::GlobalRelocalizationMsg>::SharedPtr global_relocalization_pub_;
    rclcpp::Publisher<robot_msg::msg::ChassisMsg>::SharedPtr chassis_info_pub_;
    std::shared_ptr<tools::ExtendedKalmanFilter> yaw_speed_ekf_;

    bool first_yaw_speed_ = true;
    double last_time_ = 0.0; // yaw速度计算时间
    double last_yaw_ = 0.0;
    Eigen::VectorXd yaw_speed_x0{{0.0}};
    Eigen::MatrixXd yaw_speed_P0{{1.0}};
    std::queue<double> yaw_speed_q;
    double yaw_speed_sum = 0;

    const std::string bus_name_;
    can::SocketCAN socket_can_;
    std::mutex mutex_;
    realtime_tools::RealtimeBuffer<robot_msg::msg::RotateModeMsg> rotate_mode_buffer_;
    realtime_tools::RealtimeBuffer<geometry_msgs::msg::Twist> chassis_buffer_;
    realtime_tools::RealtimeBuffer<robot_msg::msg::GimbalControlMsg> gimbal_buffer_;
    std::atomic<bool> gimbal_stop_flag_{false};

    can_frame chassis_frame_;
    can_frame gimbal_frame_;
    can_frame gimbal_stop_frame_;
    can_frame path_frame_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr path_timer_;

    std::atomic<bool> lio_running_{true};
    bool lio_stop_ = false;
    bool over_wait_time_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> lio_stop_time_;
    std::atomic<double> yaw_angle_{0.0};

    robot_msg::msg::ChassisMsg chassis_msg_;

    std::atomic<int> remaining_energy_{20000};

    /* ---------------------------------- path ---------------------------------- */
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
    int path_start_x_ = 0; // dm
    int path_start_y_ = 0; // dm
    int path_size_ = 0;
    std::vector<std::pair<int, int>> path_deltas_;
    bool path_update_ = false;                 // path是否更新
    int frame_index_ = 0;                      // 计数器
    std::atomic<bool> path_write_flag_{false}; // 是否写入路径
    std::atomic<bool> path_get_flag_{false};   // 是否获取路径
    std::mutex path_mutex_;

    float target_x_ = 0.0;
    float target_y_ = 0.0;
    static uint16_t float2uint(float x, float x_min, float x_max, uint8_t bits)
    {
      float span = x_max - x_min;
      float offset = x_min;
      return static_cast<uint16_t>((x - offset) * (static_cast<float>((1 << bits) - 1)) / span);
    }
    static float uint2float(uint16_t x_int, float x_min, float x_max, uint8_t bits)
    {
      float span = x_max - x_min;
      float offset = x_min;
      return (static_cast<float>(x_int) * span / static_cast<float>((1 << bits) - 1)) + offset;
    }
  };

} // namespace sentry_communicator

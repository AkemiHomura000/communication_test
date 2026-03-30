#pragma once

#include <string>
#include <mutex>
#include <algorithm> // For std::fill

// ROS 2
#include "geometry_msgs/msg/twist.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <realtime_tools/realtime_buffer.h>

#include <atomic>
// Superpower_hardware
#include <sentry_communicator/socketcan.h>
#include <sensor_msgs/msg/imu.hpp>
#include "robot_msg/msg/referee_info_msg.hpp"
#include "robot_msg/msg/robot_hp_msg.hpp"
#include "robot_msg/msg/chassis_msg.hpp"
#include "robot_msg/msg/posture_control_msg.hpp"
#include "robot_msg/msg/posture_feedback_msg.hpp"
#include "robot_msg/msg/chassis_mode_msg.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <cmath>
#include <cstdint>
#include <chrono>
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

  private:
    bool debug_;
    void frameCallback(const can_frame &frame);
    void cmdChassisCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void postureControlCallback(const robot_msg::msg::PostureControlMsg::SharedPtr msg);
    void chassisModeCallback(const robot_msg::msg::ChassisModeMsg::SharedPtr msg);
    
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_chassis_sub_;
    rclcpp::Publisher<robot_msg::msg::RefereeInfoMsg>::SharedPtr referee_pub_;
    rclcpp::Publisher<robot_msg::msg::RobotHpMsg>::SharedPtr team_hp_pub_;
    rclcpp::Publisher<robot_msg::msg::ChassisMsg>::SharedPtr chassis_info_pub_;
    rclcpp::Publisher<robot_msg::msg::PostureFeedbackMsg>::SharedPtr posture_feedback_pub_;
    rclcpp::Subscription<robot_msg::msg::PostureControlMsg>::SharedPtr posture_control_sub_;
    rclcpp::Subscription<robot_msg::msg::ChassisModeMsg>::SharedPtr chassis_mode_sub_;

    bool first_yaw_speed_ = true;
    double last_time_ = 0.0; // yaw速度计算时间
    double last_yaw_ = 0.0;

    const std::string bus_name_;
    can::SocketCAN socket_can_;
    std::mutex mutex_;
    std::mutex posture_feedback_mutex_;
    realtime_tools::RealtimeBuffer<geometry_msgs::msg::Twist> chassis_buffer_;
    realtime_tools::RealtimeBuffer<robot_msg::msg::PostureControlMsg> posture_buffer_;
    realtime_tools::RealtimeBuffer<robot_msg::msg::ChassisModeMsg> chassis_mode_buffer_;

    can_frame chassis_frame_;
    can_frame posture_frame_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::atomic<double> yaw_angle_{0.0};

    robot_msg::msg::ChassisMsg chassis_msg_;
    robot_msg::msg::PostureFeedbackMsg posture_feedback_;
    robot_msg::msg::RefereeInfoMsg referee_msg_;
    robot_msg::msg::RobotHpMsg team_hp_msg_;

    std::atomic<int> remaining_energy_{20000};

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

#pragma once

#include <common_lib.h>
#include "Estimator.h"

#define MAXN (720000)

// =====================================================================
// LidarImuBuffer — 封装激光雷达/IMU 数据缓冲区、回调及时间同步
// =====================================================================
class LidarImuBuffer
{
public:
  LidarImuBuffer();
  ~LidarImuBuffer() = default;

  // ── 回调（由 LaserMappingNode 的 ROS 订阅者调用）────────────────
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);
  void livox_pcl_cbk   (const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg);
  void imu_cbk         (const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in);

  // ── 时间同步：返回 true 表示一帧数据已就绪 ──────────────────────
  bool syncPackages(MeasureGroup &meas);

  // ── 公开缓冲区（LaserMappingNode 需要直接访问）──────────────────
  std::deque<PointCloudXYZI::Ptr>                lidar_buffer;
  std::deque<double>                             time_buffer;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_deque;

  sensor_msgs::msg::Imu imu_last;
  sensor_msgs::msg::Imu imu_next;

  // ── 统计 / 调试 ──────────────────────────────────────────────────
  int    scan_count = 0;
  double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];

  // ── 状态标志 ─────────────────────────────────────────────────────
  bool   lose_lid    = false;
  double imu_first_time = 0.0;

private:
  // ── 内部缓冲状态 ─────────────────────────────────────────────────
  bool   lidar_pushed_ = false;
  bool   imu_pushed_   = false;
  int    frame_ct_     = 0;
  int    wait_num_     = 0;

  double timediff_imu_wrt_lidar_ = 0.0;
  bool   timediff_set_flg_       = false;
  double time_lag_IMU_wtr_lidar_ = 0.0;  // IMU 相对 LiDAR 的时间滞后（秒）

  PointCloudXYZI::Ptr ptr_con_ {new PointCloudXYZI()};

  std::mutex m_time_;

  // ── 辅助 ─────────────────────────────────────────────────────────
  void pushLidarFrame(PointCloudXYZI::Ptr ptr, double timestamp);
};

// ── 全局单例（供 LaserMappingNode 使用）──────────────────────────────
extern LidarImuBuffer *g_lidar_imu_buffer;

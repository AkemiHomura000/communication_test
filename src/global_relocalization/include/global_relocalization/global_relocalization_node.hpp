#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/gaussian_voxelmap.hpp>
#include <small_gicp/registration/registration_helper.hpp>

namespace global_relocalization
{

/**
 * @brief 全局重定位节点
 *
 * 架构说明（异步解耦 + 点云积分）：
 *
 *   ┌─────────────────┐   点云帧队列(最近N帧)   ┌──────────────────────┐
 *   │  ROS 回调线程   │ ──(cloud_queue_)──────▶ │   配准工作线程        │
 *   │  cloudCallback  │   满N帧后通知工作线程    │   registrationLoop   │
 *   └─────────────────┘                         └──────────┬───────────┘
 *                                                           │ 更新 T_map_lidar_odom_
 *                                                           ▼
 *                                               ┌──────────────────────┐
 *                                               │   TF 定时器线程      │
 *                                               │   tfTimerCallback    │
 *                                               │   固定频率持续广播   │
 *                                               └──────────────────────┘
 *
 * 点云积分：
 *   将最近 accumulate_frames 帧点云合并后再统一降采样，作为配准的 source。
 *   在走廊、空旷等特征稀疏场景中显著提升配准鲁棒性。
 *
 * 配准统计：
 *   每次配准后打印耗时、迭代次数、fitness score、平移量、roll/pitch/yaw。
 *   同时维护累计统计（总次数、收敛率、平均耗时）。
 */
class GlobalRelocalizationNode : public rclcpp::Node
{
public:
  explicit GlobalRelocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~GlobalRelocalizationNode() override;

private:
  // ── 参数 ──────────────────────────────────────────────────────────────
  std::string map_pcd_path_;          ///< 地图 PCD 文件路径（map 坐标系）
  std::string cloud_sub_topic_;       ///< 实时点云订阅话题（lidar_odom 坐标系）
  std::string map_frame_;             ///< 地图坐标系名称
  std::string lidar_odom_frame_;      ///< 里程计坐标系名称

  // 配准参数
  double downsampling_resolution_;    ///< 体素降采样分辨率 [m]
  double voxel_resolution_;           ///< VGICP 体素分辨率 [m]
  double max_correspondence_dist_;    ///< 最大匹配点对距离 [m]
  int    num_threads_;                ///< small_gicp 线程数
  int    max_iterations_;             ///< 最大迭代次数
  double relocalization_interval_;    ///< 两次配准的最小时间间隔 [s]
  bool   publish_static_tf_;          ///< 首次收敛后是否同时发布静态 TF
  double tf_publish_hz_;              ///< TF 定时器发布频率 [Hz]

  // 点云积分参数
  int    accumulate_frames_;          ///< 积分帧数（1 = 不积分，直接用单帧）
  double accumulate_voxel_size_;      ///< 积分后去重降采样分辨率 [m]

  // ── 地图预加载数据 ────────────────────────────────────────────────────
  small_gicp::PointCloud::Ptr map_cloud_;
  std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> map_kdtree_;
  std::shared_ptr<small_gicp::GaussianVoxelMap> map_voxelmap_;
  bool map_loaded_{false};

  // ── 配准结果状态（由工作线程写，TF 定时器读，mutex 保护） ────────────
  Eigen::Isometry3d T_map_lidar_odom_;   ///< 当前估计的 map←lidar_odom 变换
  rclcpp::Time      last_tf_stamp_;      ///< 上次配准帧的时间戳
  bool              has_initial_result_{false};
  mutable std::mutex state_mutex_;

  // ── 配准时间记录（节流用，工作线程内独享，无需加锁） ─────────────────
  rclcpp::Time last_registration_time_;
  bool         throttle_initialized_{false};

  // ── 点云积分队列（ROS 回调线程写，工作线程读，mutex 保护） ──────────
  struct StampedCloud {
    small_gicp::PointCloud::Ptr cloud;  ///< 已降采样的单帧点云（未估计协方差）
    rclcpp::Time                stamp;
  };
  std::deque<StampedCloud> cloud_queue_;   ///< 积分帧队列，最多保留 accumulate_frames_ 帧
  bool                     queue_ready_{false};  ///< 队列是否已积满 accumulate_frames_ 帧
  std::mutex               queue_mutex_;
  std::condition_variable  queue_cv_;

  // ── 配准累计统计（工作线程独享，无需加锁） ───────────────────────────
  struct RegistrationStats {
    uint64_t total_count{0};       ///< 总配准次数
    uint64_t converged_count{0};   ///< 收敛次数
    double   total_time_ms{0.0};   ///< 累计耗时 [ms]
    double   min_time_ms{1e9};     ///< 最短耗时 [ms]
    double   max_time_ms{0.0};     ///< 最长耗时 [ms]
  } stats_;

  // ── 工作线程 ─────────────────────────────────────────────────────────
  std::thread       worker_thread_;
  std::atomic<bool> stop_worker_{false};

  // ── ROS 接口 ──────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
  std::shared_ptr<tf2_ros::TransformBroadcaster>       tf_broadcaster_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  // ── 私有方法 ──────────────────────────────────────────────────────────
  void loadMapPCD();

  /// ROS 回调：降采样后压入积分队列，队列满则通知工作线程
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  /// 工作线程主循环：等待队列就绪，合并积分帧，执行配准，更新状态
  void registrationLoop();

  /// TF 定时器回调：以固定频率广播当前已知变换，不依赖配准是否正在进行
  void tfTimerCallback();

  void publishTransform(const rclcpp::Time & stamp);

  /// 将积分队列中所有帧合并并二次降采样，返回处理好的 source 点云（已估计协方差）
  small_gicp::PointCloud::Ptr buildAccumulatedCloud(
    const std::deque<StampedCloud> & frames) const;

  /// 将 PCL PointXYZ 点云转换为 small_gicp::PointCloud（仅坐标，不估计协方差）
  static small_gicp::PointCloud::Ptr pclToSmallGicp(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & in);

  /// 从 Isometry3d 提取 roll/pitch/yaw（ZYX 顺序，单位：度）
  static Eigen::Vector3d toRPY(const Eigen::Isometry3d & T);
};

}  // namespace global_relocalization

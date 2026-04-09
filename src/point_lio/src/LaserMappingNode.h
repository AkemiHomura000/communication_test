// =====================================================================
// LaserMappingNode.h — LaserMappingNode 类声明
// IMU 仅用于 output 模式（state_output / kf_output）
// =====================================================================
#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "common_lib.h"
#include "Estimator.h"
#include "li_initialization.h"

#include <memory>

// =====================================================================
// LaserMappingNode
// =====================================================================
class LaserMappingNode : public rclcpp::Node
{
public:
  explicit LaserMappingNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~LaserMappingNode();

  // 在 make_shared 完成后由 main() 调用，初始化需要 shared_from_this() 的资源
  void postInit();

  // 主循环体，由 main() 以 500 Hz 驱动
  void spin_once();

  // 主循环节拍监控，由 main() 每次循环调用
  void monitorLoopTick();

private:
  // ==================================================================
  //  初始化
  // ==================================================================
  void initParameters();
  void initKalmanFilter();
  void initLogFiles();
  void initSubscribers();
  void initPublishers();
  void initServiceServer();

  // ==================================================================
  //  主循环各阶段
  // ==================================================================
  void handleReset();
  void handleFirstScan();
  void downsampleAndSort();

  // 返回 true → 初始化完成可继续；false → 仍在初始化，跳过当帧
  bool tryImuInit();

  // 返回 false → 本帧用于建图，跳过后续 KF 更新
  bool buildInitMap();

  void preparePointLists();
  void runPointByPointUpdate();
  void runImuOnlyUpdate();

  // runPointByPointUpdate 的子步骤
  void alignImuToFirstPoint(bool &imu_upda_cov);
  void processImuBeforePoint(bool &imu_upda_cov);
  void propagateState();

  // ==================================================================
  //  地图 & 发布
  // ==================================================================
  void MapIncremental();
  void publishInitMap();
  void publishFrameWorld();
  void publishFrameBody();
  void publishOdometry();
  void publishTfToBaseLink();
  void publishPath();

  // ==================================================================
  //  辅助
  // ==================================================================
  template <typename T>
  void setPoseStamp(T &out) const;

  template <typename T>
  void setTwist(T &out) const;

  static void pointBodyLidarToIMU(PointType const *pi, PointType *po);

  // ==================================================================
  //  日志
  // ==================================================================
  void logRuntimeStats(double t0, double t1, double t3, double t5);
  void logPbpState();
  void dumpLioStateToLog();

  // ==================================================================
  //  监控辅助（已移至 public，此处保留注释占位）
  // ==================================================================

  // ==================================================================
  //  退出清理
  // ==================================================================
  void saveMap();

  // ==================================================================
  //  成员变量
  // ==================================================================

  // ── 状态标志 ─────────────────────────────────────────────────────
  bool flg_first_scan_ = true;
  bool flg_reset_      = false;
  bool init_map_       = false;

  // ── 封装子模块 ────────────────────────────────────────────────────
  Estimator       estimator_;       // EKF + 点云缓冲区
  LidarImuBuffer  lidar_imu_buf_;   // 传感器数据缓冲 + 回调 + 时间同步

  // ── 卡尔曼滤波初始化矩阵 ──────────────────────────────────────────
  Eigen::Matrix<double, 30, 30> P_init_output_;
  Eigen::Matrix<double, 30, 30> Q_output_;

  // ── 点云缓存 ──────────────────────────────────────────────────────
  PointCloudXYZI::Ptr feats_undistort_  {new PointCloudXYZI()};
  PointCloudXYZI::Ptr init_feats_world_ {new PointCloudXYZI()};
  PointCloudXYZI::Ptr pcl_wait_save_    {new PointCloudXYZI()};

  // ── 体素滤波器 ────────────────────────────────────────────────────
  pcl::VoxelGrid<PointType> downSizeFilterSurf_;

  // ── ROS 消息缓存 ──────────────────────────────────────────────────
  nav_msgs::msg::Path     path_;
  nav_msgs::msg::Odometry odom_;

  // ── TF ────────────────────────────────────────────────────────────
  std::unique_ptr<tf2_ros::TransformBroadcaster>  tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer>                tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>     tf_listener_;
  geometry_msgs::msg::TransformStamped            tf_livox_to_base_;
  bool                                            tf_livox_to_base_acquired_ = false;

  // ── 发布者 ────────────────────────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_registered_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr       pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr           pub_path_;

  // ── 订阅者 ────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr     sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr             sub_imu_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr                 srv_reset_;

  // ── 性能统计 ──────────────────────────────────────────────────────
  int    frame_num_        = 0;
  int    time_log_counter_ = 0;
  double aver_time_consu_  = 0, aver_time_icp_    = 0;
  double aver_time_match_  = 0, aver_time_solve_  = 0;
  double aver_time_propag_ = 0;
  double match_time_  = 0, solve_time_  = 0;
  double propag_time_ = 0, update_time_ = 0;

  // ── 调试文件句柄 ──────────────────────────────────────────────────
  FILE *fp_ = nullptr;

  // ── 里程计发布频率控制 ────────────────────────────────────────────
  double odom_pub_interval_  = 0.0;  // 秒，0 表示每次调用均发布；由 initParameters() 设置
  double last_odom_pub_time_ = 0.0;  // 上次发布时的 time_current

  // ── spin_once 耗时统计（1 Hz 输出）───────────────────────────────
  std::chrono::steady_clock::time_point spin_last_log_time_ {std::chrono::steady_clock::now()};
  int    spin_frame_count_   = 0;
  double spin_total_ms_      = 0.0;
  double spin_downsample_ms_ = 0.0;
  double spin_preprocess_ms_ = 0.0;
  double spin_kf_update_ms_  = 0.0;

  // ── 500 Hz 主循环节拍统计 ─────────────────────────────────────────
  std::chrono::steady_clock::time_point main_loop_last_tp_ {std::chrono::steady_clock::now()};
  double main_loop_dt_sum_ms_  = 0.0;
  double main_loop_dt_max_ms_  = 0.0;
  int    main_loop_tick_count_ = 0;

  // ── 点云帧间隔监控 ────────────────────────────────────────────────
  double last_lidar_stamp_   = -1.0;   // 上一帧点云时间戳（秒）
  double lidar_interval_sum_ = 0.0;
  double lidar_interval_max_ = 0.0;
  int    lidar_interval_cnt_ = 0;

  // ── IMU 帧间隔监控 ────────────────────────────────────────────────
  double last_imu_stamp_   = -1.0;
  double imu_interval_sum_ = 0.0;
  double imu_interval_max_ = 0.0;
  int    imu_interval_cnt_ = 0;
};

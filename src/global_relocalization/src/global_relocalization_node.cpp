/**
 * @file global_relocalization_node.cpp
 * @brief 全局重定位节点。
 *
 * 工作流程：
 *  1. 启动时静态加载 PCD 地图文件（map 坐标系）。
 *  2. 订阅实时点云话题（lidar_odom 坐标系）。
 *  3. 维护点云积分队列（最近 N 帧），每次配准将 N 帧合并后再去重降采样，
 *     作为 source 与地图（target）执行 GICP，提高稀疏场景下的配准鲁棒性。
 *  4. 发布配准结果对应的 TF 变换：map → lidar_odom，并输出详细统计信息。
 *
 * ── 异步架构（3 线程解耦）────────────────────────────────────────────
 *
 *  ROS 回调线程   →  降采样/格式转换，将单帧压入积分队列，队列满则通知工作线程
 *  配准工作线程   →  等待队列积满，合并帧，执行 GICP，更新 T_map_lidar_odom_
 *  TF 定时器      →  以固定频率（默认 20 Hz）持续广播最新已知变换，
 *                    与配准耗时完全解耦，下游 TF 查询不会超时
 */

#include "global_relocalization/global_relocalization_node.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_eigen/tf2_eigen.hpp>

#include <small_gicp/util/normal_estimation_omp.hpp>

namespace global_relocalization
{

// ============================================================
//  构造函数
// ============================================================
GlobalRelocalizationNode::GlobalRelocalizationNode(const rclcpp::NodeOptions & options)
: Node("global_relocalization_node", options),
  T_map_lidar_odom_(Eigen::Isometry3d::Identity())
{
  // ── 声明并读取参数 ────────────────────────────────────────
  map_pcd_path_ = this->declare_parameter<std::string>(
    "map_pcd_path", "");
  cloud_sub_topic_ = this->declare_parameter<std::string>(
    "cloud_sub_topic", "/cloud_registered");
  map_frame_ = this->declare_parameter<std::string>(
    "map_frame", "map");
  lidar_odom_frame_ = this->declare_parameter<std::string>(
    "lidar_odom_frame", "lidar_odom");

  downsampling_resolution_ = this->declare_parameter<double>(
    "downsampling_resolution", 0.5);
  voxel_resolution_ = this->declare_parameter<double>(
    "voxel_resolution", 1.0);
  max_correspondence_dist_ = this->declare_parameter<double>(
    "max_correspondence_distance", 1.5);
  num_threads_ = this->declare_parameter<int>(
    "num_threads", 4);
  max_iterations_ = this->declare_parameter<int>(
    "max_iterations", 30);
  relocalization_interval_ = this->declare_parameter<double>(
    "relocalization_interval_sec", 0.5);
  publish_static_tf_ = this->declare_parameter<bool>(
    "publish_static_tf_after_convergence", false);
  tf_publish_hz_ = this->declare_parameter<double>(
    "tf_publish_hz", 20.0);

  // 点云积分参数
  accumulate_frames_ = this->declare_parameter<int>(
    "accumulate_frames", 5);
  accumulate_voxel_size_ = this->declare_parameter<double>(
    "accumulate_voxel_size", 0.3);

  // 参数校正
  if (accumulate_frames_ < 1) {
    RCLCPP_WARN(get_logger(),
      "accumulate_frames=%d is invalid, clamped to 1 (no accumulation).", accumulate_frames_);
    accumulate_frames_ = 1;
  }

  // ── TF 广播器 ────────────────────────────────────────────
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

  // ── 加载地图 ─────────────────────────────────────────────
  loadMapPCD();

  // ── 启动配准工作线程 ──────────────────────────────────────
  worker_thread_ = std::thread(&GlobalRelocalizationNode::registrationLoop, this);

  // ── TF 定时器：以固定频率持续广播已知变换 ────────────────
  auto period_ms = std::chrono::milliseconds(
    static_cast<int>(1000.0 / tf_publish_hz_));
  tf_timer_ = this->create_wall_timer(
    period_ms,
    std::bind(&GlobalRelocalizationNode::tfTimerCallback, this));

  // ── 订阅实时点云 ─────────────────────────────────────────
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    cloud_sub_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&GlobalRelocalizationNode::cloudCallback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(),
    "GlobalRelocalizationNode ready. topic='%s', TF=%.1fHz, "
    "accumulate_frames=%d, accumulate_voxel=%.2fm.",
    cloud_sub_topic_.c_str(), tf_publish_hz_,
    accumulate_frames_, accumulate_voxel_size_);
}

// ============================================================
//  析构函数：通知工作线程退出并等待其结束
// ============================================================
GlobalRelocalizationNode::~GlobalRelocalizationNode()
{
  stop_worker_ = true;
  queue_cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

// ============================================================
//  加载地图 PCD 文件
// ============================================================
void GlobalRelocalizationNode::loadMapPCD()
{
  if (map_pcd_path_.empty()) {
    RCLCPP_ERROR(get_logger(),
      "Parameter 'map_pcd_path' is empty. Please provide a valid PCD file path.");
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_map(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd_path_, *raw_map) == -1) {
    RCLCPP_ERROR(get_logger(),
      "Failed to load PCD file: %s", map_pcd_path_.c_str());
    return;
  }
  RCLCPP_INFO(get_logger(),
    "Loaded map PCD '%s': %zu points.", map_pcd_path_.c_str(), raw_map->size());

  // 体素降采样
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_map(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setLeafSize(
    static_cast<float>(downsampling_resolution_),
    static_cast<float>(downsampling_resolution_),
    static_cast<float>(downsampling_resolution_));
  vg.setInputCloud(raw_map);
  vg.filter(*filtered_map);
  RCLCPP_INFO(get_logger(),
    "Map after voxel downsampling: %zu points.", filtered_map->size());

  // 转换为 small_gicp::PointCloud
  map_cloud_ = pclToSmallGicp(filtered_map);

  // 估计法向量与协方差（GICP 所需）
  small_gicp::estimate_covariances_omp(*map_cloud_, 10, num_threads_);

  // 在地图上建立 KdTree
  map_kdtree_ = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
    map_cloud_, small_gicp::KdTreeBuilderOMP(num_threads_));

  // 构建高斯体素地图（供 VGICP 使用）
  map_voxelmap_ = small_gicp::create_gaussian_voxelmap(*map_cloud_, voxel_resolution_);

  map_loaded_ = true;
  RCLCPP_INFO(get_logger(), "Map preprocessing complete.");
}

// ============================================================
//  点云回调（ROS spin 线程）
//  只做降采样 + 格式转换，将单帧压入积分队列，队列满则通知工作线程。
//  不在此处估计协方差（耗时较长），留给工作线程在合并后统一处理。
// ============================================================
void GlobalRelocalizationNode::cloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!map_loaded_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Map not loaded yet, skipping cloud.");
    return;
  }

  // ROS 消息 → PCL 点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *pcl_cloud);
  if (pcl_cloud->empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Received empty point cloud, skipping.");
    return;
  }

  // 体素降采样（粗降采样，后续合并后还会再次精细降采样）
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setLeafSize(
    static_cast<float>(downsampling_resolution_),
    static_cast<float>(downsampling_resolution_),
    static_cast<float>(downsampling_resolution_));
  vg.setInputCloud(pcl_cloud);
  vg.filter(*filtered);
  if (filtered->empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "Source cloud empty after downsampling, skipping.");
    return;
  }

  // 仅转换格式，不估计协方差（节省回调线程时间）
  auto source = pclToSmallGicp(filtered);

  // 压入积分队列，超出窗口则弹出最旧帧
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    cloud_queue_.push_back({source, msg->header.stamp});
    while (static_cast<int>(cloud_queue_.size()) > accumulate_frames_) {
      cloud_queue_.pop_front();
    }
    queue_ready_ = (static_cast<int>(cloud_queue_.size()) >= accumulate_frames_);
  }

  if (queue_ready_) {
    queue_cv_.notify_one();
  }
}

// ============================================================
//  合并积分队列中所有帧 → 精细降采样 → 估计协方差
//  返回可直接送入 GICP 的 source 点云。
// ============================================================
small_gicp::PointCloud::Ptr GlobalRelocalizationNode::buildAccumulatedCloud(
  const std::deque<StampedCloud> & frames) const
{
  // 1. 将所有帧的点拼接到 PCL 点云（借助 PCL 做 VoxelGrid 去重）
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
  merged->reserve(frames.size() * frames.front().cloud->size());

  for (const auto & f : frames) {
    for (std::size_t i = 0; i < f.cloud->size(); ++i) {
      const auto & pt = f.cloud->point(i);
      merged->push_back({
        static_cast<float>(pt.x()),
        static_cast<float>(pt.y()),
        static_cast<float>(pt.z())});
    }
  }

  // 2. 精细降采样去除重叠点
  pcl::PointCloud<pcl::PointXYZ>::Ptr deduped(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setLeafSize(
    static_cast<float>(accumulate_voxel_size_),
    static_cast<float>(accumulate_voxel_size_),
    static_cast<float>(accumulate_voxel_size_));
  vg.setInputCloud(merged);
  vg.filter(*deduped);

  // 3. 转换格式并估计协方差（GICP 必须）
  auto out = pclToSmallGicp(deduped);
  small_gicp::estimate_covariances_omp(*out, 10, num_threads_);
  return out;
}

// ============================================================
//  配准工作线程主循环
//  等待积分队列满 → 复制队列 → 合并帧 → GICP → 更新状态 → 打印统计
// ============================================================
void GlobalRelocalizationNode::registrationLoop()
{
  while (!stop_worker_) {
    // 等待积分队列积满，或收到退出信号
    std::deque<StampedCloud> frames;
    rclcpp::Time latest_stamp;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [this] { return queue_ready_ || stop_worker_; });
      if (stop_worker_) break;
      frames = cloud_queue_;       // 快照：复制整个队列（小对象指针，很快）
      latest_stamp = frames.back().stamp;
      queue_ready_ = false;        // 重置，等待下一批积满后再触发
    }

    // 节流：距上次配准不足间隔则跳过
    if (throttle_initialized_) {
      double dt = (latest_stamp - last_registration_time_).seconds();
      if (dt < relocalization_interval_) {
        continue;
      }
    }

    // 取当前变换作为初始值
    Eigen::Isometry3d init_T;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      init_T = T_map_lidar_odom_;
    }

    // ── 合并积分帧 → source 点云 ──────────────────────────
    auto source = buildAccumulatedCloud(frames);

    // ── 执行 GICP 配准（耗时操作，在此线程进行） ──────────
    small_gicp::RegistrationSetting setting;
    setting.type = small_gicp::RegistrationSetting::GICP;
    setting.max_correspondence_distance = max_correspondence_dist_;
    setting.num_threads = num_threads_;
    setting.max_iterations = max_iterations_;

    const auto t0 = std::chrono::steady_clock::now();

    // target = 地图（map 系），source = 积分点云（lidar_odom 系）
    // T_target_source 即 T_map_lidar_odom
    small_gicp::RegistrationResult result =
      small_gicp::align(*map_cloud_, *source, *map_kdtree_, init_T, setting);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    last_registration_time_ = latest_stamp;
    throttle_initialized_ = true;

    // ── 更新累计统计 ──────────────────────────────────────
    stats_.total_count++;
    stats_.total_time_ms += elapsed_ms;
    if (elapsed_ms < stats_.min_time_ms) stats_.min_time_ms = elapsed_ms;
    if (elapsed_ms > stats_.max_time_ms) stats_.max_time_ms = elapsed_ms;
    if (result.converged) stats_.converged_count++;

    const double avg_time_ms =
      stats_.total_time_ms / static_cast<double>(stats_.total_count);
    const double converge_rate =
      100.0 * static_cast<double>(stats_.converged_count) /
      static_cast<double>(stats_.total_count);

    // ── 未收敛分支 ────────────────────────────────────────
    if (!result.converged) {
      RCLCPP_WARN(get_logger(),
        "[GICP #%zu] NOT converged | iters=%zu | time=%.1f ms "
        "| avg=%.1f ms | converge_rate=%.1f%%",
        stats_.total_count, result.iterations, elapsed_ms,
        avg_time_ms, converge_rate);
      continue;
    }

    // ── 提取平移和旋转 ────────────────────────────────────
    const Eigen::Vector3d t = result.T_target_source.translation();
    const Eigen::Vector3d rpy = toRPY(result.T_target_source);

    // fitness score：使用归一化误差（result.error / 点数，越小越好）
    // small_gicp 的 result.error 是 GICP 目标函数值（协方差加权残差之和）
    const double fitness = result.error / static_cast<double>(source->size());

    RCLCPP_INFO(get_logger(),
      "[GICP #%zu] converged | iters=%zu | time=%.1f ms (min=%.1f max=%.1f avg=%.1f) "
      "| converge_rate=%.1f%% | fitness=%.4f\n"
      "            translation : x=%.4f  y=%.4f  z=%.4f  [m]\n"
      "            rotation    : roll=%.3f  pitch=%.3f  yaw=%.3f  [deg]\n"
      "            source_pts=%zu  accumulated_frames=%d",
      stats_.total_count,
      result.iterations,
      elapsed_ms, stats_.min_time_ms, stats_.max_time_ms, avg_time_ms,
      converge_rate,
      fitness,
      t.x(), t.y(), t.z(),
      rpy.x(), rpy.y(), rpy.z(),
      source->size(), static_cast<int>(frames.size()));

    // ── 更新状态（TF 定时器下次触发时读取） ──────────────
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      T_map_lidar_odom_  = result.T_target_source;
      last_tf_stamp_     = latest_stamp;
      has_initial_result_ = true;
    }

    // 首次收敛：可选发布静态 TF（在定时器中处理）
    if (publish_static_tf_ && stats_.converged_count == 1) {
      RCLCPP_INFO(get_logger(),
        "First convergence achieved. Static TF will be published by timer.");
    }
  }
}

// ============================================================
//  TF 定时器回调（以 tf_publish_hz_ 固定频率触发）
// ============================================================
void GlobalRelocalizationNode::tfTimerCallback()
{
  publishTransform(this->now());
}

// ============================================================
//  发布 TF 变换：map → lidar_odom
// ============================================================
void GlobalRelocalizationNode::publishTransform(const rclcpp::Time & stamp)
{
  Eigen::Isometry3d T;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    if (!has_initial_result_) return;
    T = T_map_lidar_odom_;
  }

  geometry_msgs::msg::TransformStamped ts = tf2::eigenToTransform(T);
  ts.header.stamp    = stamp;
  ts.header.frame_id = map_frame_;         // 父坐标系：map
  ts.child_frame_id  = lidar_odom_frame_;  // 子坐标系：lidar_odom

  tf_broadcaster_->sendTransform(ts);

  if (publish_static_tf_) {
    static_tf_broadcaster_->sendTransform(ts);
  }
}

// ============================================================
//  PCL PointXYZ → small_gicp::PointCloud 转换（仅坐标，不估计协方差）
// ============================================================
small_gicp::PointCloud::Ptr GlobalRelocalizationNode::pclToSmallGicp(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & in)
{
  auto out = std::make_shared<small_gicp::PointCloud>();
  out->resize(in->size());
  for (std::size_t i = 0; i < in->size(); ++i) {
    out->point(i) = Eigen::Vector4d(
      in->points[i].x,
      in->points[i].y,
      in->points[i].z,
      1.0);
  }
  return out;
}

// ============================================================
//  从 Isometry3d 提取 roll / pitch / yaw（ZYX 欧拉角，单位：度）
//  eulerAngles(2,1,0) → [yaw, pitch, roll]，反转后 → [roll, pitch, yaw]
// ============================================================
Eigen::Vector3d GlobalRelocalizationNode::toRPY(const Eigen::Isometry3d & T)
{
  // ZYX 顺序：eulerAngles(2,1,0) 返回 [yaw, pitch, roll]（弧度）
  Eigen::Vector3d ypr = T.rotation().eulerAngles(2, 1, 0);
  constexpr double RAD2DEG = 180.0 / M_PI;
  return Eigen::Vector3d(ypr.z() * RAD2DEG,   // roll
                         ypr.y() * RAD2DEG,   // pitch
                         ypr.x() * RAD2DEG);  // yaw
}

}  // namespace global_relocalization

// ============================================================
//  main 入口
// ============================================================
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<global_relocalization::GlobalRelocalizationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

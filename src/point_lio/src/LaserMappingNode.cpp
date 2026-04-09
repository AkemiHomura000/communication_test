// =====================================================================
// LaserMappingNode.cpp — LaserMappingNode class implementation
// =====================================================================

#include "LaserMappingNode.h"
#include "parameters.h"

#include <malloc.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ament_index_cpp/get_package_prefix.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <chrono>

using namespace std;

// =====================================================================
//  Constructor / Destructor
// =====================================================================
LaserMappingNode::LaserMappingNode(const rclcpp::NodeOptions &options)
    : Node("laserMapping", options)
{
  // Set global singleton pointer for esekf callback bridging
  g_estimator        = &estimator_;
  g_lidar_imu_buffer = &lidar_imu_buf_;

  initParameters();
  initKalmanFilter();
  initLogFiles();
  initSubscribers();
  initPublishers();   // 不含 TF（需要 shared_from_this）
  initServiceServer();
  std::memset(estimator_.point_selected_surf, true,
              sizeof(estimator_.point_selected_surf));
  // Note: postInit() must be called explicitly by main() after make_shared completes
}

LaserMappingNode::~LaserMappingNode()
{
  saveMap();
  fout_out.close();
  fout_imu_pbp.close();
  if (fp_)
  {
    fclose(fp_);
    fp_ = nullptr;
  }
  RCLCPP_INFO(this->get_logger(), "pointlio finished");
}

// =====================================================================
//  postInit — call after shared_ptr is ready
// =====================================================================
void LaserMappingNode::postInit()
{
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(shared_from_this());
  tf_buffer_      = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  RCLCPP_INFO(this->get_logger(), "LaserMappingNode initialized");
}

// =====================================================================
//  Main loop body
// =====================================================================
void LaserMappingNode::spin_once()
{
  if (!lidar_imu_buf_.syncPackages(Measures))
    return;

  if (flg_reset_)
    handleReset();

  if (flg_first_scan_)
    handleFirstScan();

  // performance timing
  const double t0 = omp_get_wtime();
  match_time_ = solve_time_ = propag_time_ = update_time_ = 0.0;

  // downsampling
  const double t1 = omp_get_wtime();
  downsampleAndSort();

  // IMU gravity initialization
  if (!p_imu->after_imu_init_)
  {
    if (!tryImuInit())
      return;
  }

  // Initial map building
  if (!init_map_)
  {
    if (!buildInitMap())
      return;
  }

  // Precompute point cross-matrix
  preparePointLists();

  const double t3 = omp_get_wtime();

  // Kalman iterative update
  if (!estimator_.time_seq.empty())
    runPointByPointUpdate();
  else
    runImuOnlyUpdate();

  // Incremental map update
  if (estimator_.feats_down_size > 4)
    MapIncremental();

  const double t5 = omp_get_wtime();

  // 话题发布
  if (path_en)
    publishPath();
  if (scan_pub_en || pcd_save_en)
    publishFrameWorld();
  if (scan_pub_en && scan_body_pub_en)
    publishFrameBody();

  // ── Timing statistics (1 Hz output) ──────────────────────────────────────────
  {
    const double total_ms       = (t5 - t0)  * 1000.0;
    const double downsample_ms  = (t1 - t0)  * 1000.0;
    const double preprocess_ms  = (t3 - t1)  * 1000.0;
    const double kf_update_ms   = (t5 - t3)  * 1000.0;

    // Warn if processing time exceeds lidar frame period (default ~100 ms)
    const double lidar_period_ms = (lidar_interval_cnt_ > 0)
        ? (lidar_interval_sum_ / lidar_interval_cnt_ * 1000.0)
        : 100.0;
    if (total_ms > lidar_period_ms)
      RCLCPP_WARN(this->get_logger(),
                  "[Perf] frame processing %.2f ms exceeds lidar period %.2f ms, real-time violated!",
                  total_ms, lidar_period_ms);

    spin_frame_count_++;
    spin_total_ms_      += total_ms;
    spin_downsample_ms_ += downsample_ms;
    spin_preprocess_ms_ += preprocess_ms;
    spin_kf_update_ms_  += kf_update_ms;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(now - spin_last_log_time_).count();
    if (elapsed_s >= 1.0)
    {
      const double n = static_cast<double>(spin_frame_count_);

      // LiDAR and IMU queue depths
      const int lidar_q  = static_cast<int>(lidar_imu_buf_.lidar_buffer.size());
      const int imu_q    = static_cast<int>(lidar_imu_buf_.imu_deque.size());

      // LiDAR frame interval stats
      const double lidar_avg_ms = (lidar_interval_cnt_ > 0)
          ? (lidar_interval_sum_ / lidar_interval_cnt_ * 1000.0) : 0.0;
      const double lidar_max_ms = lidar_interval_max_ * 1000.0;

      // IMU frame interval stats
      const double imu_avg_ms = (imu_interval_cnt_ > 0)
          ? (imu_interval_sum_ / imu_interval_cnt_ * 1000.0) : 0.0;
      const double imu_max_ms = imu_interval_max_ * 1000.0;

      // Main loop tick stats
      const double loop_avg_ms = (main_loop_tick_count_ > 0)
          ? (main_loop_dt_sum_ms_ / main_loop_tick_count_) : 0.0;
      const double loop_max_ms = main_loop_dt_max_ms_;
      const double loop_actual_hz = (loop_avg_ms > 0.0) ? (1000.0 / loop_avg_ms) : 0.0;

      RCLCPP_INFO(this->get_logger(),
                  "\n"
                  "─────────────── 1s Stats ───────────────\n"
                  " Frames: %d  │  Raw pts: %d  │  Downsampled: %d pts\n"
                  " Time(avg): total=%.2fms  ds=%.2fms  pre=%.2fms  kf=%.2fms\n"
                  " LiDAR queue: %d  │  avg_dt: %.2fms  max_dt: %.2fms\n"
                  " IMU   queue: %d  │  avg_dt: %.2fms  max_dt: %.2fms\n"
                  " Loop tick:  avg=%.3fms (%.1fHz)  max=%.3fms  (target 500Hz=2.0ms)\n"
                  "────────────────────────────────────────",
                  spin_frame_count_,
                  static_cast<int>(feats_undistort_->size()),
                  estimator_.feats_down_size,
                  spin_total_ms_ / n,
                  spin_downsample_ms_ / n,
                  spin_preprocess_ms_ / n,
                  spin_kf_update_ms_  / n,
                  lidar_q, lidar_avg_ms, lidar_max_ms,
                  imu_q,   imu_avg_ms,   imu_max_ms,
                  loop_avg_ms, loop_actual_hz, loop_max_ms);

      // Reset stats
      spin_frame_count_   = 0;
      spin_total_ms_      = 0.0;
      spin_downsample_ms_ = 0.0;
      spin_preprocess_ms_ = 0.0;
      spin_kf_update_ms_  = 0.0;
      spin_last_log_time_ = now;

      lidar_interval_sum_ = 0.0; lidar_interval_max_ = 0.0; lidar_interval_cnt_ = 0;
      imu_interval_sum_   = 0.0; imu_interval_max_   = 0.0; imu_interval_cnt_   = 0;
      main_loop_dt_sum_ms_ = 0.0; main_loop_dt_max_ms_ = 0.0; main_loop_tick_count_ = 0;
    }
  }

  // Debug stats (detailed file log)
  if (runtime_pos_log)
    logRuntimeStats(t0, t1, t3, t5);
}

// =====================================================================
//  Initialization
// =====================================================================
void LaserMappingNode::initParameters()
{
  auto shared_this = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
  readParameters(shared_this);
  RCLCPP_INFO(this->get_logger(), "lidar_type: %d", lidar_type);

  estimator_.ivox_ = std::make_shared<IVoxType>(ivox_options_);

  downSizeFilterSurf_.setLeafSize(
      filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

  estimator_.Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  estimator_.Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  if (extrinsic_est_en)
  {
    estimator_.kf_output.x_.offset_R_L_I = estimator_.Lidar_R_wrt_IMU;
    estimator_.kf_output.x_.offset_T_L_I = estimator_.Lidar_T_wrt_IMU;
  }

  p_imu->lidar_type = p_pre->lidar_type = lidar_type;
  p_imu->imu_en     = imu_en;

  // Compute odometry publish minimum interval based on configured frequency (0 = publish every time)
  odom_pub_interval_ = (odom_pub_freq > 0.0) ? (1.0 / odom_pub_freq) : 0.0;
  RCLCPP_INFO(this->get_logger(), "odom_pub_freq: %.1f Hz (interval: %.4f s)",
              odom_pub_freq, odom_pub_interval_);

  path_.header.stamp    = get_ros_time(lidar_end_time);
  path_.header.frame_id = "lidar_odom";
}

void LaserMappingNode::initKalmanFilter()
{
  reset_cov_output(P_init_output_);
  Q_output_ = Estimator::process_noise_cov_output();
  estimator_.initKalmanFilter(P_init_output_, Q_output_);
}

void LaserMappingNode::initLogFiles()
{
  string pos_log_dir = string(ROOT_DIR) + "/Log/pos_log.txt";
  fp_ = fopen(pos_log_dir.c_str(), "w");
  open_file();
}

void LaserMappingNode::initSubscribers()
{
  // Custom QoS: BEST_EFFORT (match sensor publishers), large depth to absorb scheduling jitter
  auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(20)).best_effort().durability_volatile();
  auto imu_qos   = rclcpp::QoS(rclcpp::KeepLast(200)).best_effort().durability_volatile();

  if (p_pre->lidar_type == AVIA)
  {
    sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        lid_topic, lidar_qos,
        [this](const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
        {
          lidar_imu_buf_.livox_pcl_cbk(msg);
          const double t = rclcpp::Time(msg->header.stamp).seconds();
          if (last_lidar_stamp_ > 0.0)
          {
            const double dt = t - last_lidar_stamp_;
            lidar_interval_sum_ += dt;
            lidar_interval_cnt_++;
            if (dt > lidar_interval_max_) lidar_interval_max_ = dt;
            if (dt > 0.15)
              RCLCPP_WARN(this->get_logger(),
                          "[LiDAR] large frame gap: %.3f s (expected ~0.1s)", dt);
          }
          last_lidar_stamp_ = t;
        });
  }
  else
  {
    sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lid_topic, lidar_qos,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
          lidar_imu_buf_.standard_pcl_cbk(msg);
          const double t = rclcpp::Time(msg->header.stamp).seconds();
          if (last_lidar_stamp_ > 0.0)
          {
            const double dt = t - last_lidar_stamp_;
            lidar_interval_sum_ += dt;
            lidar_interval_cnt_++;
            if (dt > lidar_interval_max_) lidar_interval_max_ = dt;
            if (dt > 0.15)
              RCLCPP_WARN(this->get_logger(),
                          "[LiDAR] large frame gap: %.3f s (expected ~0.1s)", dt);
          }
          last_lidar_stamp_ = t;
        });
  }
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, imu_qos,
      [this](const sensor_msgs::msg::Imu::ConstSharedPtr &msg)
      {
        lidar_imu_buf_.imu_cbk(msg);
        const double t = rclcpp::Time(msg->header.stamp).seconds();
        if (last_imu_stamp_ > 0.0)
        {
          const double dt = t - last_imu_stamp_;
          imu_interval_sum_ += dt;
          imu_interval_cnt_++;
          if (dt > imu_interval_max_) imu_interval_max_ = dt;
          if (dt > 0.02)   // IMU typically 200 Hz, normal interval ~5 ms
            RCLCPP_WARN(this->get_logger(),
                        "[IMU] large frame gap: %.4f s (expected ~0.005s)", dt);
        }
        last_imu_stamp_ = t;
      });
}

void LaserMappingNode::initPublishers()
{
  pub_cloud_registered_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
  pub_cloud_body_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 1000);
  pub_laser_map_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 1000);
  pub_odom_ =
      this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", rclcpp::SensorDataQoS());
  pub_path_ =
      this->create_publisher<nav_msgs::msg::Path>("/path", 1000);
  // tf_broadcaster_ / tf_buffer_ / tf_listener_ are initialized in postInit()
}

void LaserMappingNode::initServiceServer()
{
  srv_reset_ = this->create_service<std_srvs::srv::Trigger>(
      "/reset_map_and_odom",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res)
      {
        extern bool g_flg_exit;
        g_flg_exit   = true;
        res->success = true;
        res->message = "stop signal sent.";
      });
}

// =====================================================================
//  Reset handling
// =====================================================================
void LaserMappingNode::handleReset()
{
  RCLCPP_WARN(this->get_logger(), "reset when rosbag play back");
  p_imu->Reset();
  feats_undistort_.reset(new PointCloudXYZI());
  state_out = state_output();
  estimator_.kf_output.change_P(P_init_output_);
  flg_first_scan_ = true;
  is_first_frame  = true;
  flg_reset_      = false;
  init_map_       = false;
  estimator_.ivox_.reset(new IVoxType(ivox_options_));
}

// =====================================================================
//  First scan handling
// =====================================================================
void LaserMappingNode::handleFirstScan()
{
  first_lidar_time = Measures.lidar_beg_time;
  flg_first_scan_  = false;

  if (first_imu_time < 1.0)
  {
    first_imu_time = get_time_sec(lidar_imu_buf_.imu_next.header.stamp);
    RCLCPP_INFO(this->get_logger(), "first imu time: %f", first_imu_time);
  }
  time_current = 0.0;

  if (imu_en)
  {
    estimator_.kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
    while (Measures.lidar_beg_time > get_time_sec(lidar_imu_buf_.imu_next.header.stamp))
    {
      lidar_imu_buf_.imu_deque.pop_front();
      if (lidar_imu_buf_.imu_deque.empty())
        break;
      lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
      lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
    }
  }
  else
  {
    estimator_.kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
    estimator_.kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
    estimator_.kf_output.x_.acc *= -1;
    p_imu->imu_need_init_ = false;
  }
  estimator_.G_m_s2 = std::sqrt(gravity[0] * gravity[0] +
                                 gravity[1] * gravity[1] +
                                 gravity[2] * gravity[2]);
}

// =====================================================================
//  Pointcloud downsampling and sorting
// =====================================================================
void LaserMappingNode::downsampleAndSort()
{
  p_imu->Process(Measures, feats_undistort_);
  if (space_down_sample)
  {
    downSizeFilterSurf_.setInputCloud(feats_undistort_);
    downSizeFilterSurf_.filter(*estimator_.feats_down_body);
  }
  else
  {
    estimator_.feats_down_body = Measures.lidar;
  }
  sort(estimator_.feats_down_body->points.begin(),
       estimator_.feats_down_body->points.end(), time_list);
  estimator_.time_seq        = time_compressing<int>(estimator_.feats_down_body);
  estimator_.feats_down_size = estimator_.feats_down_body->points.size();
}

// =====================================================================
//  IMU gravity direction initialization
// =====================================================================
bool LaserMappingNode::tryImuInit()
{
  if (p_imu->imu_need_init_)
    return false;

  V3D tmp_gravity;
  if (imu_en)
  {
    tmp_gravity = -p_imu->mean_acc / p_imu->mean_acc.norm() * estimator_.G_m_s2;
  }
  else
  {
    tmp_gravity << VEC_FROM_ARRAY(gravity_init);
    p_imu->after_imu_init_ = true;
  }
  M3D rot_init;
  p_imu->Set_init(tmp_gravity, rot_init);
  estimator_.kf_output.x_.rot = rot_init;
  estimator_.kf_output.x_.acc =
      -rot_init.transpose() * estimator_.kf_output.x_.gravity;
  return true;
}

// =====================================================================
//  Initial map building
// =====================================================================
bool LaserMappingNode::buildInitMap()
{
  estimator_.feats_down_world->resize(feats_undistort_->size());
  for (size_t i = 0; i < feats_undistort_->size(); i++)
    estimator_.pointBodyToWorld(&feats_undistort_->points[i],
                                &estimator_.feats_down_world->points[i]);

  for (size_t i = 0; i < estimator_.feats_down_world->size(); i++)
    init_feats_world_->points.emplace_back(estimator_.feats_down_world->points[i]);

  if (init_feats_world_->size() < static_cast<size_t>(init_map_size))
    return false;

  estimator_.ivox_->AddPoints(init_feats_world_->points);
  publishInitMap();
  RCLCPP_INFO(this->get_logger(), "initial map size: %zu", init_feats_world_->size());
  init_feats_world_.reset(new PointCloudXYZI());
  init_map_ = true;
  return false; // frame used for map building, skip subsequent KF update
}

// =====================================================================
//  Precompute cross-matrix list
// =====================================================================
void LaserMappingNode::preparePointLists()
{
  estimator_.normvec->resize(estimator_.feats_down_size);
  estimator_.feats_down_world->resize(estimator_.feats_down_size);
  estimator_.Nearest_Points.resize(estimator_.feats_down_size);
  estimator_.crossmat_list.reserve(estimator_.feats_down_size);
  estimator_.pbody_list.reserve(estimator_.feats_down_size);

  for (size_t i = 0; i < estimator_.feats_down_body->size(); i++)
  {
    V3D point_this(estimator_.feats_down_body->points[i].x,
                   estimator_.feats_down_body->points[i].y,
                   estimator_.feats_down_body->points[i].z);
    estimator_.pbody_list[i] = point_this;
    if (!extrinsic_est_en)
    {
      point_this = estimator_.Lidar_R_wrt_IMU * point_this + estimator_.Lidar_T_wrt_IMU;
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);
      estimator_.crossmat_list[i] = point_crossmat;
    }
  }
}

// =====================================================================
//  Point-by-point IMU propagation + Kalman update
// =====================================================================
void LaserMappingNode::runPointByPointUpdate()
{
  const double pcl_beg_time = Measures.lidar_beg_time;
  estimator_.idx               = -1;
  bool imu_upda_cov = false;

  for (estimator_.k = 0;
       estimator_.k < static_cast<int>(estimator_.time_seq.size());
       estimator_.k++)
  {
    PointType &point_body =
        estimator_.feats_down_body->points[estimator_.idx + estimator_.time_seq[estimator_.k]];
    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

    if (is_first_frame)
      alignImuToFirstPoint(imu_upda_cov);

    if (imu_en && !lidar_imu_buf_.imu_deque.empty())
      processImuBeforePoint(imu_upda_cov);

    if (flg_reset_)
      break;

    propagateState();

    double t_update_start = omp_get_wtime();

    if (estimator_.feats_down_size < 1)
    {
      RCLCPP_WARN(this->get_logger(), "No point, skip this scan!");
      estimator_.idx += estimator_.time_seq[estimator_.k];
      continue;
    }

    if (!estimator_.kf_output.update_iterated_dyn_share_modified())
    {
      estimator_.idx += estimator_.time_seq[estimator_.k];
      continue;
    }

    double solve_start = omp_get_wtime();

    // Rate control: publish odometry at configured odom_pub_freq during point-by-point update
    if (odom_pub_interval_ <= 0.0 ||
        (time_current - last_odom_pub_time_) >= odom_pub_interval_)
    {
      publishOdometry();
      last_odom_pub_time_ = time_current;
      if (runtime_pos_log)
        logPbpState();
    }

    for (int j = 0; j < estimator_.time_seq[estimator_.k]; j++)
    {
      PointType &pb = estimator_.feats_down_body->points[estimator_.idx + j + 1];
      PointType &pw = estimator_.feats_down_world->points[estimator_.idx + j + 1];
      estimator_.pointBodyToWorld(&pb, &pw);
    }

    solve_time_  += omp_get_wtime() - solve_start;
    update_time_ += omp_get_wtime() - t_update_start;
    estimator_.idx += estimator_.time_seq[estimator_.k];
  }
}

// =====================================================================
//  No laser points: propagate state with IMU only
// =====================================================================
void LaserMappingNode::runImuOnlyUpdate()
{
  if (lidar_imu_buf_.imu_deque.empty())
    return;

  lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
  lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());

  while (get_time_sec(lidar_imu_buf_.imu_next.header.stamp) > time_current &&
         get_time_sec(lidar_imu_buf_.imu_next.header.stamp) <
             Measures.lidar_beg_time + lidar_time_inte)
  {
    if (is_first_frame)
    {
      while (get_time_sec(lidar_imu_buf_.imu_next.header.stamp) <
             Measures.lidar_beg_time + lidar_time_inte)
      {
        lidar_imu_buf_.imu_deque.pop_front();
        if (lidar_imu_buf_.imu_deque.empty())
          break;
        lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
        lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
      }
      break;
    }

    time_current = get_time_sec(lidar_imu_buf_.imu_next.header.stamp);

    double dt_cov = time_current - time_update_last;
    if (dt_cov > 0.0)
    {
      estimator_.kf_output.predict(dt_cov, Q_output_, estimator_.input_in, false, true);
      time_update_last = time_current;
    }

    double dt = time_current - time_predict_last_const;
    estimator_.kf_output.predict(dt, Q_output_, estimator_.input_in, true, false);
    time_predict_last_const = time_current;

    estimator_.angvel_avr << lidar_imu_buf_.imu_next.angular_velocity.x,
                              lidar_imu_buf_.imu_next.angular_velocity.y,
                              lidar_imu_buf_.imu_next.angular_velocity.z;
    estimator_.acc_avr    << lidar_imu_buf_.imu_next.linear_acceleration.x,
                              lidar_imu_buf_.imu_next.linear_acceleration.y,
                              lidar_imu_buf_.imu_next.linear_acceleration.z;
    estimator_.kf_output.update_iterated_dyn_share_IMU();

    lidar_imu_buf_.imu_deque.pop_front();
    if (lidar_imu_buf_.imu_deque.empty())
      break;
    lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
    lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
  }
}

// =====================================================================
//  Align IMU time to the first laser point
// =====================================================================
void LaserMappingNode::alignImuToFirstPoint(bool &imu_upda_cov)
{
  if (imu_en)
  {
    while (time_current > get_time_sec(lidar_imu_buf_.imu_next.header.stamp))
    {
      lidar_imu_buf_.imu_deque.pop_front();
      if (lidar_imu_buf_.imu_deque.empty())
        break;
      lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
      lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
    }
    estimator_.angvel_avr << lidar_imu_buf_.imu_last.angular_velocity.x,
                              lidar_imu_buf_.imu_last.angular_velocity.y,
                              lidar_imu_buf_.imu_last.angular_velocity.z;
    estimator_.acc_avr    << lidar_imu_buf_.imu_last.linear_acceleration.x,
                              lidar_imu_buf_.imu_last.linear_acceleration.y,
                              lidar_imu_buf_.imu_last.linear_acceleration.z;
  }
  is_first_frame          = false;
  imu_upda_cov            = true;
  time_update_last        = time_current;
  time_predict_last_const = time_current;
}

// =====================================================================
//  Process IMU data prior to the current laser point timestamp
// =====================================================================
void LaserMappingNode::processImuBeforePoint(bool &imu_upda_cov)
{
  bool last_imu =
      (get_time_sec(lidar_imu_buf_.imu_next.header.stamp) ==
       get_time_sec(lidar_imu_buf_.imu_deque.front()->header.stamp));
  while (get_time_sec(lidar_imu_buf_.imu_next.header.stamp) < time_predict_last_const &&
         !lidar_imu_buf_.imu_deque.empty())
  {
    if (!last_imu)
    {
      lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
      lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
      break;
    }
    lidar_imu_buf_.imu_deque.pop_front();
    if (lidar_imu_buf_.imu_deque.empty())
      break;
    lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
    lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
  }

  while (time_current > get_time_sec(lidar_imu_buf_.imu_next.header.stamp))
  {
    imu_upda_cov = true;
    estimator_.angvel_avr << lidar_imu_buf_.imu_next.angular_velocity.x,
                              lidar_imu_buf_.imu_next.angular_velocity.y,
                              lidar_imu_buf_.imu_next.angular_velocity.z;
    estimator_.acc_avr    << lidar_imu_buf_.imu_next.linear_acceleration.x,
                              lidar_imu_buf_.imu_next.linear_acceleration.y,
                              lidar_imu_buf_.imu_next.linear_acceleration.z;

    double dt = get_time_sec(lidar_imu_buf_.imu_next.header.stamp) - time_predict_last_const;
    estimator_.kf_output.predict(dt, Q_output_, estimator_.input_in, true, false);
    time_predict_last_const = get_time_sec(lidar_imu_buf_.imu_next.header.stamp);

    double dt_cov = get_time_sec(lidar_imu_buf_.imu_next.header.stamp) - time_update_last;
    if (dt_cov > 0.0)
    {
      time_update_last = get_time_sec(lidar_imu_buf_.imu_next.header.stamp);
      double t0 = omp_get_wtime();
      estimator_.kf_output.predict(dt_cov, Q_output_, estimator_.input_in, false, true);
      propag_time_ += omp_get_wtime() - t0;

      double t1 = omp_get_wtime();
      estimator_.kf_output.update_iterated_dyn_share_IMU();
      solve_time_ += omp_get_wtime() - t1;
    }

    lidar_imu_buf_.imu_deque.pop_front();
    if (lidar_imu_buf_.imu_deque.empty())
      break;
    lidar_imu_buf_.imu_last = lidar_imu_buf_.imu_next;
    lidar_imu_buf_.imu_next = *(lidar_imu_buf_.imu_deque.front());
  }
}

// =====================================================================
//  Propagate state to the current laser point timestamp
// =====================================================================
void LaserMappingNode::propagateState()
{
  double dt    = time_current - time_predict_last_const;
  double t_beg = omp_get_wtime();

  if (!prop_at_freq_of_imu)
  {
    double dt_cov = time_current - time_update_last;
    if (dt_cov > 0.0)
    {
      estimator_.kf_output.predict(dt_cov, Q_output_, estimator_.input_in, false, true);
      time_update_last = time_current;
    }
  }
  estimator_.kf_output.predict(dt, Q_output_, estimator_.input_in, true, false);
  propag_time_ += omp_get_wtime() - t_beg;
  time_predict_last_const = time_current;
}

// =====================================================================
//  Incremental map update
// =====================================================================
void LaserMappingNode::MapIncremental()
{
  PointVector points_to_add;
  const int cur_pts = estimator_.feats_down_world->size();
  points_to_add.reserve(cur_pts);

  for (int i = 0; i < cur_pts; ++i)
  {
    PointType &pw = estimator_.feats_down_world->points[i];
    if (!estimator_.Nearest_Points[i].empty())
    {
      const PointVector &near = estimator_.Nearest_Points[i];
      Eigen::Vector3f center =
          ((pw.getVector3fMap() / filter_size_map_min).array().floor() + 0.5f) *
          filter_size_map_min;

      bool need_add = true;
      for (const auto &np : near)
      {
        Eigen::Vector3f d = np.getVector3fMap() - center;
        if (fabs(d.x()) < 0.5f * filter_size_map_min &&
            fabs(d.y()) < 0.5f * filter_size_map_min &&
            fabs(d.z()) < 0.5f * filter_size_map_min)
        {
          need_add = false;
          break;
        }
      }
      if (need_add)
        points_to_add.emplace_back(pw);
    }
    else
    {
      points_to_add.emplace_back(pw);
    }
  }
  estimator_.ivox_->AddPoints(points_to_add);
}

// =====================================================================
//  Publish: initial map pointcloud
// =====================================================================
void LaserMappingNode::publishInitMap()
{
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*init_feats_world_, msg);
  msg.header.stamp    = get_ros_time(lidar_end_time);
  msg.header.frame_id = "lidar_odom";
  pub_laser_map_->publish(msg);
}

// =====================================================================
//  Publish: world-frame pointcloud
// =====================================================================
void LaserMappingNode::publishFrameWorld()
{
  if (scan_pub_en)
  {
    const int size = estimator_.feats_down_world->points.size();
    PointCloudXYZI::Ptr cloud_world(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
      cloud_world->points[i] = estimator_.feats_down_world->points[i];

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud_world, msg);
    msg.header.stamp    = get_ros_time(lidar_end_time);
    msg.header.frame_id = "lidar_odom";
    pub_cloud_registered_->publish(msg);
  }

  if (pcd_save_en)
    *pcl_wait_save_ += *estimator_.feats_down_world;
}

// =====================================================================
//  Publish: body-frame pointcloud
// =====================================================================
void LaserMappingNode::publishFrameBody()
{
  const int size = feats_undistort_->points.size();
  PointCloudXYZI::Ptr cloud_body(new PointCloudXYZI(size, 1));
  for (int i = 0; i < size; i++)
    pointBodyLidarToIMU(&feats_undistort_->points[i], &cloud_body->points[i]);

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud_body, msg);
  msg.header.stamp    = get_ros_time(lidar_end_time);
  msg.header.frame_id = "livox_frame";
  pub_cloud_body_->publish(msg);
}
void LaserMappingNode::publishOdometry()
{
  odom_.header.frame_id = "lidar_odom";
  odom_.header.stamp    = get_ros_time(time_current);

  // ---- 1. Look up fixed livox_frame -> base_link extrinsic (only once) ----
  if (!tf_livox_to_base_acquired_)
  {
    try
    {
      tf_livox_to_base_ = tf_buffer_->lookupTransform(
          "livox_frame", "base_link", odom_.header.stamp);
      tf_livox_to_base_acquired_ = true;
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Failed to lookup livox_frame -> base_link: %s", ex.what());
      // Fallback when extrinsic not yet available: publish livox_frame quantities
      odom_.child_frame_id = "livox_frame";
      setPoseStamp(odom_.pose.pose);
      setTwist(odom_.twist.twist);
      pub_odom_->publish(odom_);
      return;
    }
  }

  // ---- 2. Compute T_odom_base = T_odom_livox * T_livox_base ----
  tf2::Transform T_odom_livox, T_livox_base;
  {
    geometry_msgs::msg::Pose livox_pose;
    setPoseStamp(livox_pose);
    tf2::fromMsg(livox_pose, T_odom_livox);
  }
  tf2::fromMsg(tf_livox_to_base_.transform, T_livox_base);
  tf2::Transform T_odom_base = T_odom_livox * T_livox_base;

  // ---- 3. Fill Pose (base_link pose in odom frame) ----
  odom_.child_frame_id = "base_link";
  tf2::toMsg(T_odom_base, odom_.pose.pose);

  // ---- 4. Fill Twist (base_link velocity in odom frame) ----
  // linear vel: v_base = v_livox + ω_odom × r_livox_to_base_in_odom
  //   where r_livox_to_base_in_odom = R_odom_livox * t_livox_to_base
  {
    // R_odom_livox (livox orientation in world frame)
    Eigen::Quaterniond q_odom_livox(estimator_.kf_output.x_.rot);
    Eigen::Matrix3d R_odom_livox = q_odom_livox.toRotationMatrix();

    // t_livox_to_base (vector from livox origin to base_link origin, in livox frame)
    const auto &t = tf_livox_to_base_.transform.translation;
    Eigen::Vector3d t_livox_base(t.x, t.y, t.z);

    // r expressed in odom frame
    Eigen::Vector3d r_odom = R_odom_livox * t_livox_base;

    // ω in odom frame (recomputed directly here)
    V3D omg_world = R_odom_livox * estimator_.kf_output.x_.omg;

    // v_livox in odom frame
    V3D v_livox = estimator_.kf_output.x_.vel;

    // v_base = v_livox + ω × r
    Eigen::Vector3d v_base = v_livox + omg_world.cross(r_odom);

    odom_.twist.twist.linear.x  = v_base(0);
    odom_.twist.twist.linear.y  = v_base(1);
    odom_.twist.twist.linear.z  = v_base(2);

    // angular velocity: same for rigid body, expressed in odom frame
    odom_.twist.twist.angular.x = omg_world(0);
    odom_.twist.twist.angular.y = omg_world(1);
    odom_.twist.twist.angular.z = omg_world(2);
  }

  pub_odom_->publish(odom_);
  publishTfToBaseLink();
}

void LaserMappingNode::publishTfToBaseLink()
{
  if (!tf_livox_to_base_acquired_)
    return;  // 外参未就绪，TF 暂不广播

  tf2::Transform T_odom_livox, T_livox_base;
  {
    geometry_msgs::msg::Pose livox_pose;
    setPoseStamp(livox_pose);
    tf2::fromMsg(livox_pose, T_odom_livox);
  }
  tf2::fromMsg(tf_livox_to_base_.transform, T_livox_base);

  geometry_msgs::msg::TransformStamped ts;
  ts.header.stamp    = odom_.header.stamp;
  ts.header.frame_id = "lidar_odom";
  ts.child_frame_id  = "base_link";
  ts.transform       = tf2::toMsg(T_odom_livox * T_livox_base);
  tf_broadcaster_->sendTransform(ts);
}

// =====================================================================
//  发布：轨迹
// =====================================================================
void LaserMappingNode::publishPath()
{
  geometry_msgs::msg::PoseStamped pose_stamped;
  setPoseStamp(pose_stamped.pose);
  pose_stamped.header.stamp    = get_ros_time(lidar_end_time);
  pose_stamped.header.frame_id = "lidar_odom";
  path_.poses.emplace_back(pose_stamped);
  pub_path_->publish(path_);
}

// =====================================================================
//  辅助：从 kf_output 填充 ROS Pose / Twist（模板显式实例化）
// =====================================================================
template <typename T>
void LaserMappingNode::setPoseStamp(T &out) const
{
  out.position.x = estimator_.kf_output.x_.pos(0);
  out.position.y = estimator_.kf_output.x_.pos(1);
  out.position.z = estimator_.kf_output.x_.pos(2);
  Eigen::Quaterniond q(estimator_.kf_output.x_.rot);
  out.orientation.x = q.coeffs()[0];
  out.orientation.y = q.coeffs()[1];
  out.orientation.z = q.coeffs()[2];
  out.orientation.w = q.coeffs()[3];
}

template <typename T>
void LaserMappingNode::setTwist(T &out) const
{
  out.linear.x  = estimator_.kf_output.x_.vel(0);
  out.linear.y  = estimator_.kf_output.x_.vel(1);
  out.linear.z  = estimator_.kf_output.x_.vel(2);
  // omg 是 IMU body 系角速度，旋转到 lidar_odom 世界系：ω_world = R * ω_body
  Eigen::Quaterniond q_rot(estimator_.kf_output.x_.rot);
  V3D omg_world = q_rot.toRotationMatrix() * estimator_.kf_output.x_.omg;
  out.angular.x = omg_world(0);
  out.angular.y = omg_world(1);
  out.angular.z = omg_world(2);
}

// 显式实例化，避免链接错误
template void LaserMappingNode::setPoseStamp(geometry_msgs::msg::Pose &) const;
template void LaserMappingNode::setTwist(geometry_msgs::msg::Twist &) const;

// =====================================================================
//  辅助：LiDAR body → IMU body 坐标变换
// =====================================================================
void LaserMappingNode::pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
  V3D p_lidar(pi->x, pi->y, pi->z);
  V3D p_imu = extrinsic_est_en
      ? (g_estimator->kf_output.x_.offset_R_L_I * p_lidar +
         g_estimator->kf_output.x_.offset_T_L_I)
      : (g_estimator->Lidar_R_wrt_IMU * p_lidar + g_estimator->Lidar_T_wrt_IMU);
  po->x         = p_imu(0);
  po->y         = p_imu(1);
  po->z         = p_imu(2);
  po->intensity = pi->intensity;
}

// =====================================================================
//  运行时性能统计
// =====================================================================
void LaserMappingNode::logRuntimeStats(double t0, double t1, double t3, double t5)
{
  frame_num_++;
  auto update_avg = [&](double &avg, double val)
  { avg = avg * (frame_num_ - 1) / frame_num_ + val / frame_num_; };

  update_avg(aver_time_consu_,  t5 - t0);
  update_avg(aver_time_icp_,    update_time_);
  update_avg(aver_time_match_,  match_time_);
  update_avg(aver_time_solve_,  solve_time_);
  update_avg(aver_time_propag_, propag_time_);

  lidar_imu_buf_.T1[lidar_imu_buf_.scan_count]     = Measures.lidar_beg_time;
  lidar_imu_buf_.s_plot[time_log_counter_]          = t5 - t0;
  lidar_imu_buf_.s_plot2[time_log_counter_]         = feats_undistort_->points.size();
  lidar_imu_buf_.s_plot3[time_log_counter_]         = aver_time_consu_;
  time_log_counter_++;

  printf("[ mapping ] IMU+DS: %.6f  match: %.6f  solve: %.6f  "
         "ICP: %.6f  map_incre: %.6f  total: %.6f  propag: %.6f\n",
         t1 - t0, aver_time_match_, aver_time_solve_,
         t3 - t1, t5 - t3, aver_time_consu_, aver_time_propag_);

  {
    V3D euler = SO3ToEuler(estimator_.kf_output.x_.rot);
    fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time
             << " " << euler.transpose()
             << " " << estimator_.kf_output.x_.pos.transpose()
             << " " << estimator_.kf_output.x_.vel.transpose()
             << " " << estimator_.kf_output.x_.omg.transpose()
             << " " << estimator_.kf_output.x_.acc.transpose()
             << " " << estimator_.kf_output.x_.gravity.transpose()
             << " " << estimator_.kf_output.x_.bg.transpose()
             << " " << estimator_.kf_output.x_.ba.transpose()
             << " " << feats_undistort_->points.size() << endl;
  }
  dumpLioStateToLog();
}

void LaserMappingNode::logPbpState()
{
  V3D euler = SO3ToEuler(estimator_.kf_output.x_.rot);
  fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time
           << " " << euler.transpose()
           << " " << estimator_.kf_output.x_.pos.transpose()
           << " " << estimator_.kf_output.x_.vel.transpose()
           << " " << estimator_.kf_output.x_.omg.transpose()
           << " " << estimator_.kf_output.x_.acc.transpose()
           << " " << estimator_.kf_output.x_.gravity.transpose()
           << " " << estimator_.kf_output.x_.bg.transpose()
           << " " << estimator_.kf_output.x_.ba.transpose()
           << " " << feats_undistort_->points.size() << endl;
}

void LaserMappingNode::dumpLioStateToLog()
{
  if (!fp_)
    return;
  V3D ang = SO3ToEuler(estimator_.kf_output.x_.rot);
  fprintf(fp_, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp_, "%lf %lf %lf ", ang(0), ang(1), ang(2));
  fprintf(fp_, "%lf %lf %lf ",
          estimator_.kf_output.x_.pos(0),
          estimator_.kf_output.x_.pos(1),
          estimator_.kf_output.x_.pos(2));
  fprintf(fp_, "%lf %lf %lf ", 0.0, 0.0, 0.0);
  fprintf(fp_, "%lf %lf %lf ",
          estimator_.kf_output.x_.vel(0),
          estimator_.kf_output.x_.vel(1),
          estimator_.kf_output.x_.vel(2));
  fprintf(fp_, "%lf %lf %lf ", 0.0, 0.0, 0.0);
  fprintf(fp_, "%lf %lf %lf ",
          estimator_.kf_output.x_.bg(0),
          estimator_.kf_output.x_.bg(1),
          estimator_.kf_output.x_.bg(2));
  fprintf(fp_, "%lf %lf %lf ",
          estimator_.kf_output.x_.ba(0),
          estimator_.kf_output.x_.ba(1),
          estimator_.kf_output.x_.ba(2));
  fprintf(fp_, "%lf %lf %lf ",
          estimator_.kf_output.x_.gravity(0),
          estimator_.kf_output.x_.gravity(1),
          estimator_.kf_output.x_.gravity(2));
  fprintf(fp_, "\r\n");
  fflush(fp_);
}

// =====================================================================
//  退出时保存地图
// =====================================================================
void LaserMappingNode::saveMap()
{
  if (pcd_save_en && pcl_wait_save_->size() > 0)
  {
    string path = string(ROOT_DIR) + "PCD/map.pcd";
    pcl::PCDWriter writer;
    writer.writeBinary(path, *pcl_wait_save_);
    RCLCPP_INFO(this->get_logger(), "pcd saved to %s", path.c_str());
  }
}

// =====================================================================
//  主循环节拍监控（由 main() 每次 while 循环调用）
// =====================================================================
void LaserMappingNode::monitorLoopTick()
{
  const auto now = std::chrono::steady_clock::now();
  const double dt_ms =
      std::chrono::duration<double, std::milli>(now - main_loop_last_tp_).count();
  main_loop_last_tp_ = now;

  if (dt_ms <= 0.0 || dt_ms > 100.0)   // 跳过第一次及异常值
    return;

  main_loop_dt_sum_ms_ += dt_ms;
  main_loop_tick_count_++;
  if (dt_ms > main_loop_dt_max_ms_)
    main_loop_dt_max_ms_ = dt_ms;

  // 500 Hz 目标周期为 2 ms，超过 4 ms（2 倍）报 warning
  if (dt_ms > 50.0)
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 200,
                         "[MainLoop] tick overrun: %.3f ms (target 2.0 ms / 500 Hz)", dt_ms);
}

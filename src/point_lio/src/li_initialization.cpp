#include "li_initialization.h"

// ── 全局单例（由 LaserMappingNode 构造时设置）────────────────────────
LidarImuBuffer *g_lidar_imu_buffer = nullptr;

// =====================================================================
//  构造函数
// =====================================================================
LidarImuBuffer::LidarImuBuffer()
{
  std::memset(T1,       0, sizeof(T1));
  std::memset(s_plot,   0, sizeof(s_plot));
  std::memset(s_plot2,  0, sizeof(s_plot2));
  std::memset(s_plot3,  0, sizeof(s_plot3));
  std::memset(s_plot11, 0, sizeof(s_plot11));
}

// =====================================================================
//  内部辅助：将一帧点云压入缓冲
// =====================================================================
void LidarImuBuffer::pushLidarFrame(PointCloudXYZI::Ptr ptr, double timestamp)
{
  if (con_frame)
  {
    if (frame_ct_ == 0)
      time_con = last_timestamp_lidar;

    if (frame_ct_ < 10)
    {
      for (auto &pt : ptr->points)
        pt.curvature += (last_timestamp_lidar - time_con) * 1000.0f;
      for (const auto &pt : ptr->points)
        ptr_con_->push_back(pt);
      frame_ct_++;
    }
    else
    {
      PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
      *ptr_con_i = *ptr_con_;
      lidar_buffer.push_back(ptr_con_i);
      time_buffer.push_back(time_con);
      ptr_con_->clear();
      frame_ct_ = 0;
    }
  }
  else
  {
    lidar_buffer.emplace_back(ptr);
    time_buffer.emplace_back(timestamp);
  }
}

// =====================================================================
//  标准 PointCloud2 回调
// =====================================================================
void LidarImuBuffer::standard_pcl_cbk(
    const sensor_msgs::msg::PointCloud2::SharedPtr &msg)
{
  scan_count++;
  const double preprocess_start_time = omp_get_wtime();

  const double t_stamp = rclcpp::Time(msg->header.stamp).seconds();
  if (t_stamp < last_timestamp_lidar)
  {
    RCLCPP_ERROR(rclcpp::get_logger("LidarImuBuffer"),
                 "lidar loop back, clear buffer");
    return;
  }
  last_timestamp_lidar = t_stamp;

  if ((lidar_type == VELO16 || lidar_type == OUST64 || lidar_type == HESAIxt32) &&
      cut_frame_init)
  {
    std::deque<PointCloudXYZI::Ptr> ptr;
    std::deque<double> timestamp_lidar;
    p_pre->process_cut_frame_pcl2(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);
    while (!ptr.empty() && !timestamp_lidar.empty())
    {
      lidar_buffer.push_back(ptr.front());
      ptr.pop_front();
      time_buffer.push_back(timestamp_lidar.front() / 1000.0);
      timestamp_lidar.pop_front();
    }
  }
  else
  {
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI(20000, 1));
    p_pre->process(msg, ptr);
    pushLidarFrame(ptr, t_stamp);
  }
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
}

// =====================================================================
//  Livox CustomMsg 回调
// =====================================================================
void LidarImuBuffer::livox_pcl_cbk(
    const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg)
{
  scan_count++;
  const double preprocess_start_time = omp_get_wtime();

  const double t_stamp = rclcpp::Time(msg->header.stamp).seconds();
  if (t_stamp < last_timestamp_lidar)
  {
    RCLCPP_ERROR(rclcpp::get_logger("LidarImuBuffer"),
                 "lidar loop back, clear buffer");
    return;
  }
  last_timestamp_lidar = t_stamp;

  if (cut_frame_init)
  {
    std::deque<PointCloudXYZI::Ptr> ptr;
    std::deque<double> timestamp_lidar;
    p_pre->process_cut_frame_livox(msg, ptr, timestamp_lidar, cut_frame_num, scan_count);
    while (!ptr.empty() && !timestamp_lidar.empty())
    {
      lidar_buffer.push_back(ptr.front());
      ptr.pop_front();
      time_buffer.push_back(timestamp_lidar.front() / 1000.0);
      timestamp_lidar.pop_front();
    }
  }
  else
  {
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI(10000, 1));
    p_pre->process(msg, ptr);
    pushLidarFrame(ptr, t_stamp);
  }
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
}

// =====================================================================
//  IMU 回调
// =====================================================================
void LidarImuBuffer::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in)
{
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
  msg->header.stamp = get_ros_time(
      get_time_sec(msg_in->header.stamp) - timediff_imu_wrt_lidar_ - time_lag_IMU_wtr_lidar_);

  const double timestamp = get_time_sec(msg->header.stamp);
  if (timestamp < last_timestamp_imu)
  {
    RCLCPP_ERROR(rclcpp::get_logger("LidarImuBuffer"),
                 "imu loop back, clear deque");
    return;
  }
  imu_deque.emplace_back(msg);
  last_timestamp_imu = timestamp;
}

// =====================================================================
//  时间同步：返回 true 表示一帧 LiDAR+IMU 数据就绪
// =====================================================================
bool LidarImuBuffer::syncPackages(MeasureGroup &meas)
{
  if (!imu_en)
  {
    // ── 无 IMU 模式 ────────────────────────────────────────────────
    if (lidar_buffer.empty())
      return false;

    if (!lidar_pushed_)
    {
      meas.lidar          = lidar_buffer.front();
      meas.lidar_beg_time = time_buffer.front();
      lose_lid = false;

      if (meas.lidar->points.empty())
      {
        std::cout << "lose lidar" << std::endl;
        lose_lid = true;
      }
      else
      {
        double end_time = meas.lidar->points.back().curvature;
        for (const auto &pt : meas.lidar->points)
          if (pt.curvature > end_time)
            end_time = pt.curvature;
        lidar_end_time          = meas.lidar_beg_time + end_time / 1000.0;
        meas.lidar_last_time    = lidar_end_time;
      }
      lidar_pushed_ = true;
    }

    time_buffer.pop_front();
    lidar_buffer.pop_front();
    lidar_pushed_ = false;
    return !lose_lid;
  }

  // ── 有 IMU 模式 ──────────────────────────────────────────────────
  if (lidar_buffer.empty() || imu_deque.empty())
    return false;

  if (!lidar_pushed_)
  {
    lose_lid = false;
    meas.lidar          = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();

    if (meas.lidar->points.empty())
    {
      std::cout << "lose lidar" << std::endl;
      lose_lid = true;
    }
    else
    {
      double end_time = meas.lidar->points.back().curvature;
      for (const auto &pt : meas.lidar->points)
        if (pt.curvature > end_time)
          end_time = pt.curvature;
      lidar_end_time       = meas.lidar_beg_time + end_time / 1000.0;
      meas.lidar_last_time = lidar_end_time;
    }
    lidar_pushed_ = true;
  }

  if (!lose_lid && (last_timestamp_imu < lidar_end_time))
    return false;
  if (lose_lid && last_timestamp_imu < meas.lidar_beg_time + lidar_time_inte)
    return false;

  if (!lose_lid && !imu_pushed_)
  {
    if (p_imu->imu_need_init_)
    {
      double imu_time = get_time_sec(imu_deque.front()->header.stamp);
      imu_next = *(imu_deque.front());
      meas.imu.shrink_to_fit();
      while (imu_time < lidar_end_time)
      {
        meas.imu.emplace_back(imu_deque.front());
        imu_last = imu_next;
        imu_deque.pop_front();
        if (imu_deque.empty()) break;
        imu_time = get_time_sec(imu_deque.front()->header.stamp);
        imu_next = *(imu_deque.front());
      }
    }
    imu_pushed_ = true;
  }

  if (lose_lid && !imu_pushed_)
  {
    if (p_imu->imu_need_init_)
    {
      double imu_time = get_time_sec(imu_deque.front()->header.stamp);
      meas.imu.shrink_to_fit();
      imu_next = *(imu_deque.front());
      while (imu_time < meas.lidar_beg_time + lidar_time_inte)
      {
        meas.imu.emplace_back(imu_deque.front());
        imu_last = imu_next;
        imu_deque.pop_front();
        if (imu_deque.empty()) break;
        imu_time = get_time_sec(imu_deque.front()->header.stamp);
        imu_next = *(imu_deque.front());
      }
    }
    imu_pushed_ = true;
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed_ = false;
  imu_pushed_   = false;
  return true;
}

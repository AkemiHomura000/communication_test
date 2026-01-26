#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

// Patchwork++-ROS
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

#include "GroundSegmentationServer.hpp"
#include "Utils.hpp"

namespace patchworkpp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

GroundSegmentationServer::GroundSegmentationServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("patchworkpp_node", options) {
  patchwork::Params params;
  base_frame_  = declare_parameter<std::string>("base_frame", base_frame_);
  window_size_ = declare_parameter<int>("window_size", window_size_);
  voxel_size_  = declare_parameter<double>("voxel_size", voxel_size_);

  params.sensor_height = declare_parameter<double>("sensor_height", params.sensor_height);

  params.num_iter    = declare_parameter<int>("num_iter", params.num_iter);
  params.num_lpr     = declare_parameter<int>("num_lpr", params.num_lpr);
  params.num_min_pts = declare_parameter<int>("num_min_pts", params.num_min_pts);
  params.th_seeds    = declare_parameter<double>("th_seeds", params.th_seeds);

  params.th_dist    = declare_parameter<double>("th_dist", params.th_dist);
  params.th_seeds_v = declare_parameter<double>("th_seeds_v", params.th_seeds_v);
  params.th_dist_v  = declare_parameter<double>("th_dist_v", params.th_dist_v);

  params.max_range       = declare_parameter<double>("max_range", params.max_range);
  params.min_range       = declare_parameter<double>("min_range", params.min_range);
  params.uprightness_thr = declare_parameter<double>("uprightness_thr", params.uprightness_thr);

  params.verbose = get_parameter<bool>("verbose", params.verbose);

  // ToDo. Support intensity
  params.enable_RNR = false;

  // Construct the main Patchwork++ node
  Patchworkpp_ = std::make_unique<patchwork::PatchWorkpp>(params);

  // Initialize subscribers
  pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered",
      rclcpp::SensorDataQoS(),
      std::bind(&GroundSegmentationServer::EstimateGround, this, std::placeholders::_1));

  /*
   * We use the following QoS setting for reliable ground segmentation.
   * If you want to run Patchwork++ in real-time and real-world operation,
   * please change the QoS setting
   */
  //  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  cloud_publisher_  = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/cloud", qos);
  ground_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ground", qos);
  nonground_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonground", qos);

  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

void GroundSegmentationServer::EstimateGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  const Eigen::MatrixXf &cloud = patchworkpp_ros::utils::PointCloud2ToEigenMat(msg);

  cloud_buffer_.push_back(cloud);
  if (cloud_buffer_.size() > (size_t)window_size_) {
    cloud_buffer_.pop_front();
  }

  Eigen::MatrixX3f accumulated_cloud;
  if (cloud_buffer_.size() > 1) {
    long total_rows = 0;
    for (const auto &c : cloud_buffer_) {
      total_rows += c.rows();
    }
    accumulated_cloud.resize(total_rows, 3);
    long current_row = 0;
    for (const auto &c : cloud_buffer_) {
      accumulated_cloud.block(current_row, 0, c.rows(), 3) = c;
      current_row += c.rows();
    }
  } else {
    accumulated_cloud = cloud;
  }
  RCLCPP_INFO(this->get_logger(), "Cloud buffer size: %ld", cloud_buffer_.size());
  RCLCPP_INFO(this->get_logger(), "Accumulated cloud size: %ld", accumulated_cloud.rows());
  if (voxel_size_ > 0.0) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_cloud->points.reserve(accumulated_cloud.rows());
    for (int i = 0; i < accumulated_cloud.rows(); ++i) {
      pcl_cloud->points.emplace_back(
          accumulated_cloud(i, 0), accumulated_cloud(i, 1), accumulated_cloud(i, 2));
    }

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pcl_cloud);
    vg.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    vg.filter(filtered_cloud);

    accumulated_cloud.resize(filtered_cloud.size(), 3);
    for (size_t i = 0; i < filtered_cloud.size(); ++i) {
      accumulated_cloud.row(i) << filtered_cloud.points[i].x, filtered_cloud.points[i].y,
          filtered_cloud.points[i].z;
    }
  }

  // Estimate ground
  Patchworkpp_->estimateGround(accumulated_cloud);

  cloud_publisher_->publish(
      patchworkpp_ros::utils::EigenMatToPointCloud2(accumulated_cloud, msg->header));
  // Get ground and nonground
  Eigen::MatrixX3f ground    = Patchworkpp_->getGround();
  Eigen::MatrixX3f nonground = Patchworkpp_->getNonground();
  double time_taken          = Patchworkpp_->getTimeTaken();
  PublishClouds(ground, nonground, msg->header);
}

void GroundSegmentationServer::PublishClouds(const Eigen::MatrixX3f &est_ground,
                                             const Eigen::MatrixX3f &est_nonground,
                                             const std_msgs::msg::Header header_msg) {
  std_msgs::msg::Header header = header_msg;
  header.frame_id              = base_frame_;
  ground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_ground, header)));
  nonground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_nonground, header)));
}
}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)

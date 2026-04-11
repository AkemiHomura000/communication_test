/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "small_point_lio_node.hpp"
#include "io/pcd_io.h"
#include "lidar_adapter/custom_mid360_driver.h"
#include "lidar_adapter/livox_custom_msg.h"
#include "lidar_adapter/livox_pointcloud2.h"
#include "lidar_adapter/unitree_lidar.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace small_point_lio {

    SmallPointLioNode::SmallPointLioNode(const rclcpp::NodeOptions &options)
        : Node("small_point_lio", options) {
        std::string lidar_topic = declare_parameter<std::string>("lidar_topic");
        std::string imu_topic = declare_parameter<std::string>("imu_topic");
        std::string lidar_type = declare_parameter<std::string>("lidar_type");
        std::string lidar_frame = declare_parameter<std::string>("lidar_frame");
        bool save_pcd = declare_parameter<bool>("save_pcd");
        save_pcd_ = save_pcd;
        small_point_lio = std::make_unique<small_point_lio::SmallPointLio>(*this);
        odometry_publisher = create_publisher<nav_msgs::msg::Odometry>("/Odometry", rclcpp::SensorDataQoS());
        pointcloud_publisher = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 1000);
        path_publisher = create_publisher<nav_msgs::msg::Path>("/path", 1000);
        path_msg.header.frame_id = "lidar_odom";
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
        if (save_pcd) {
            pointcloud_mapping = std::make_unique<util::PointcloudMapping>(0.02);
        }
        map_save_trigger = create_service<std_srvs::srv::Trigger>(
                "map_save",
                [this, save_pcd, lidar_frame](const std_srvs::srv::Trigger::Request::SharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res) {
                    if (!save_pcd) {
                        res->success = false;
                        res->message = "pcd save is disabled";
                        RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "pcd save is disabled");
                        return;
                    }
                    res->success = true;
                    RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "waiting for pcd saving ...");
                    auto pointcloud_to_save = std::make_shared<std::vector<Eigen::Vector3f>>();
                    *pointcloud_to_save = pointcloud_mapping->get_points();
                    bool tf_ready = pcd_tf_cached_;
                    Eigen::Matrix3f R = R_pcd_;
                    Eigen::Vector3f T = T_pcd_;
                    std::thread([pointcloud_to_save, tf_ready, R, T]() {
                        if (tf_ready) {
                            for (auto &p : *pointcloud_to_save) { p = R * p + T; }
                        }
                        io::pcd::write_pcd(ROOT_DIR + "/pcd/scan.pcd", *pointcloud_to_save);
                        RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "save pcd success");
                    }).detach();
                });
        small_point_lio->set_odometry_callback([this, lidar_frame](const common::Odometry &odometry) {
            last_odometry = odometry;

            builtin_interfaces::msg::Time time_msg;
            time_msg.sec = std::floor(odometry.timestamp);
            time_msg.nanosec = static_cast<uint32_t>((odometry.timestamp - time_msg.sec) * 1e9);

            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped.header.stamp = time_msg;
            transform_stamped.header.frame_id = "lidar_odom";
            transform_stamped.child_frame_id = "base_link";
            geometry_msgs::msg::TransformStamped base_link_to_lidar_frame_transform;
            try {
                base_link_to_lidar_frame_transform = tf_buffer->lookupTransform(lidar_frame, "base_link", time_msg);
            } catch (tf2::TransformException &ex) {
                RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "Failed to lookup transform from base_link to %s: %s", lidar_frame.c_str(), ex.what());
                return;
            }
            tf2::Transform tf_lidar_odom_to_lidar_frame;
            tf_lidar_odom_to_lidar_frame.setOrigin(tf2::Vector3(odometry.position.x(), odometry.position.y(), odometry.position.z()));
            tf_lidar_odom_to_lidar_frame.setRotation(tf2::Quaternion(odometry.orientation.x(), odometry.orientation.y(), odometry.orientation.z(), odometry.orientation.w()));
            tf2::Transform tf_base_link_to_lidar_frame;
            tf2::fromMsg(base_link_to_lidar_frame_transform.transform, tf_base_link_to_lidar_frame);
            tf2::Transform tf_odom_to_base_link = tf_base_link_to_lidar_frame.inverse() * tf_lidar_odom_to_lidar_frame * tf_base_link_to_lidar_frame;
            transform_stamped.transform = tf2::toMsg(tf_odom_to_base_link);

            nav_msgs::msg::Odometry odometry_msg;
            odometry_msg.header.stamp = time_msg;
            odometry_msg.header.frame_id = "lidar_odom";
            odometry_msg.child_frame_id = "base_link";
            odometry_msg.pose.pose.position.x = transform_stamped.transform.translation.x;
            odometry_msg.pose.pose.position.y = transform_stamped.transform.translation.y;
            odometry_msg.pose.pose.position.z = transform_stamped.transform.translation.z;
            odometry_msg.pose.pose.orientation.x = transform_stamped.transform.rotation.x;
            odometry_msg.pose.pose.orientation.y = transform_stamped.transform.rotation.y;
            odometry_msg.pose.pose.orientation.z = transform_stamped.transform.rotation.z;
            odometry_msg.pose.pose.orientation.w = transform_stamped.transform.rotation.w;

            // Express base_link velocity in lidar_odom world frame.
            // odometry.velocity         : velocity of lidar_frame(=IMU) origin, in world frame
            // odometry.angular_velocity : angular velocity in lidar_frame body frame
            // odometry.orientation      : R_{lidar->world} (lidar body -> world rotation)
            //
            // Full rotation chain:
            //   omega_world = R_{lidar->world} * omega_lidar
            //               = R_{base->world} * R_{lidar->base} * omega_lidar
            //
            // R_{lidar->base} = R_{base->lidar}^T = tf_base_link_to_lidar_frame.getBasis().transpose()
            // This correctly maps the 45-deg rolled lidar angular velocity back to base_link axes.

            tf2::Matrix3x3 R_lidar_to_world(tf2::Quaternion(
                odometry.orientation.x(), odometry.orientation.y(),
                odometry.orientation.z(), odometry.orientation.w()));

            // R_{base->lidar} from static TF, R_{lidar->base} = transpose
            tf2::Matrix3x3 R_base_to_lidar = tf_base_link_to_lidar_frame.getBasis();
            tf2::Matrix3x3 R_lidar_to_base = R_base_to_lidar.transpose();

            tf2::Vector3 omega_lidar(odometry.angular_velocity.x(), odometry.angular_velocity.y(), odometry.angular_velocity.z());
            // First rotate omega from lidar_frame to base_link frame (undoes the 45-deg roll)
            tf2::Vector3 omega_base = R_lidar_to_base * omega_lidar;
            // Then rotate from base_link frame to world frame: R_{base->world} = R_{lidar->world} * R_{lidar->base}^T
            tf2::Matrix3x3 R_base_to_world = R_lidar_to_world * R_base_to_lidar;
            tf2::Vector3 omega_world = R_base_to_world * omega_base;

            // Lever-arm compensation:
            // v_lidar_world is the velocity of lidar_frame origin (not base_link origin).
            // When rotating around base_link, lidar_frame origin moves with v = omega x r_{base->lidar}.
            // So: v_base_world = v_lidar_world - R_{lidar->world} * (omega_lidar x r_{lidar->base}^{lidar})
            //                  = v_lidar_world - R_{lidar->world} * (omega_lidar x r_{lidar->base})
            // Note: r_{lidar->base} = -r_{base->lidar}, so equivalently:
            //       v_base_world = v_lidar_world + R_{lidar->world} * (omega_lidar x r_{base->lidar})
            tf2::Vector3 r_lidar_to_base_in_lidar = tf_base_link_to_lidar_frame.inverse().getOrigin();
            tf2::Vector3 lever_world = R_lidar_to_world * omega_lidar.cross(r_lidar_to_base_in_lidar);

            tf2::Vector3 v_lidar_world(odometry.velocity.x(), odometry.velocity.y(), odometry.velocity.z());
            tf2::Vector3 v_base_world = v_lidar_world - lever_world;

            // // === DEBUG OUTPUT (remove after verification) ===
            // static int debug_count = 0;
            // if (++debug_count % 50 == 0) {
            //     RCLCPP_INFO(get_logger(), "--- velocity debug ---");
            //     RCLCPP_INFO(get_logger(), "v_lidar_world (raw):    x=%.3f y=%.3f z=%.3f  |norm|=%.3f",
            //         v_lidar_world.x(), v_lidar_world.y(), v_lidar_world.z(), v_lidar_world.length());
            //     RCLCPP_INFO(get_logger(), "r_lidar_to_base_in_lidar: x=%.3f y=%.3f z=%.3f",
            //         r_lidar_to_base_in_lidar.x(), r_lidar_to_base_in_lidar.y(), r_lidar_to_base_in_lidar.z());
            //     RCLCPP_INFO(get_logger(), "omega_lidar: x=%.3f y=%.3f z=%.3f",
            //         omega_lidar.x(), omega_lidar.y(), omega_lidar.z());
            //     RCLCPP_INFO(get_logger(), "omega_lidar x r_lidar_to_base (body): x=%.3f y=%.3f z=%.3f",
            //         omega_lidar.cross(r_lidar_to_base_in_lidar).x(),
            //         omega_lidar.cross(r_lidar_to_base_in_lidar).y(),
            //         omega_lidar.cross(r_lidar_to_base_in_lidar).z());
            //     RCLCPP_INFO(get_logger(), "lever_world:            x=%.3f y=%.3f z=%.3f",
            //         lever_world.x(), lever_world.y(), lever_world.z());
            //     RCLCPP_INFO(get_logger(), "v_base_world (result):  x=%.3f y=%.3f z=%.3f  |norm|=%.3f",
            //         v_base_world.x(), v_base_world.y(), v_base_world.z(), v_base_world.length());
            // }
            // // === END DEBUG ===

            odometry_msg.twist.twist.linear.x = v_base_world.x();
            odometry_msg.twist.twist.linear.y = v_base_world.y();
            odometry_msg.twist.twist.linear.z = v_base_world.z();
            // omega_base is angular velocity expressed in base_link body frame (correct axes, z=yaw)
            odometry_msg.twist.twist.angular.x = omega_base.x();
            odometry_msg.twist.twist.angular.y = omega_base.y();
            odometry_msg.twist.twist.angular.z = omega_base.z();

            tf_broadcaster->sendTransform(transform_stamped);
            odometry_publisher->publish(odometry_msg);

            // geometry_msgs::msg::PoseStamped pose_stamped;
            // pose_stamped.header.stamp = time_msg;
            // pose_stamped.header.frame_id = "lidar_odom";
            // pose_stamped.pose = odometry_msg.pose.pose;
            // path_msg.header.stamp = time_msg;
            // path_msg.poses.push_back(pose_stamped);
            // path_publisher->publish(path_msg);
        });
        small_point_lio->set_pointcloud_callback([this, save_pcd, lidar_frame](const std::vector<Eigen::Vector3f> &pointcloud) {
            if (pointcloud_publisher->get_subscription_count() > 0) {
                builtin_interfaces::msg::Time time_msg;
                time_msg.sec = std::floor(last_odometry.timestamp);
                time_msg.nanosec = static_cast<uint32_t>((last_odometry.timestamp - time_msg.sec) * 1e9);

                geometry_msgs::msg::TransformStamped lidar_frame_to_base_link_transform;
                try {
                    lidar_frame_to_base_link_transform = tf_buffer->lookupTransform("base_link", lidar_frame, time_msg);
                } catch (tf2::TransformException &ex) {
                    RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "Failed to lookup transform from %s to base_link: %s", lidar_frame.c_str(), ex.what());
                    return;
                }
                Eigen::Vector3f lidar_frame_to_base_link_T;
                lidar_frame_to_base_link_T << static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.x),
                        static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.y),
                        static_cast<float>(lidar_frame_to_base_link_transform.transform.translation.z);
                Eigen::Matrix3f lidar_frame_to_base_link_R =
                        Eigen::Quaternionf(
                                static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.w),
                                static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.x),
                                static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.y),
                                static_cast<float>(lidar_frame_to_base_link_transform.transform.rotation.z))
                                .toRotationMatrix();
                sensor_msgs::msg::PointCloud2 msg;
                msg.header.stamp = time_msg;
                msg.header.frame_id = "lidar_odom";
                msg.width = pointcloud.size();
                msg.height = 1;
                msg.fields.reserve(4);
                sensor_msgs::msg::PointField field;
                field.name = "x";
                field.offset = 0;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "y";
                field.offset = 4;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "z";
                field.offset = 8;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "intensity";
                field.offset = 12;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                msg.is_bigendian = false;
                msg.point_step = 16;
                msg.row_step = msg.width * msg.point_step;
                msg.data.resize(msg.row_step * msg.height);
                Eigen::Vector3f transformed_point;
                auto pointer = reinterpret_cast<float *>(msg.data.data());
                for (const auto &point: pointcloud) {
                    transformed_point = lidar_frame_to_base_link_R * point + lidar_frame_to_base_link_T;
                    *pointer = transformed_point.x();
                    ++pointer;
                    *pointer = transformed_point.y();
                    ++pointer;
                    *pointer = transformed_point.z();
                    ++pointer;
                    *pointer = 0;
                    ++pointer;
                }
                msg.is_dense = false;
                pointcloud_publisher->publish(msg);
            }
            if (save_pcd) {
                if (!pcd_tf_cached_) {
                    try {
                        auto tf = tf_buffer->lookupTransform("base_link", lidar_frame, tf2::TimePointZero);
                        T_pcd_ << static_cast<float>(tf.transform.translation.x),
                                  static_cast<float>(tf.transform.translation.y),
                                  static_cast<float>(tf.transform.translation.z);
                        R_pcd_ = Eigen::Quaternionf(
                            static_cast<float>(tf.transform.rotation.w),
                            static_cast<float>(tf.transform.rotation.x),
                            static_cast<float>(tf.transform.rotation.y),
                            static_cast<float>(tf.transform.rotation.z)).toRotationMatrix();
                        pcd_tf_cached_ = true;
                    } catch (tf2::TransformException &ex) {
                        RCLCPP_WARN_ONCE(rclcpp::get_logger("small_point_lio"), "PCD TF not yet available: %s", ex.what());
                    }
                }
                for (const auto &point: pointcloud) {
                    pointcloud_mapping->add_point(point);
                }
            }
        });
        if (lidar_type == "livox_custom_msg") {
#ifdef HAVE_LIVOX_DRIVER
            lidar_adapter = std::make_unique<LivoxCustomMsgAdapter>();
#else
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "livox_custom_msg requested but not available!");
            rclcpp::shutdown();
            return;
#endif
        } else if (lidar_type == "livox_pointcloud2") {
            lidar_adapter = std::make_unique<LivoxPointCloud2Adapter>();
        } else if (lidar_type == "custom_mid360_driver") {
            lidar_adapter = std::make_unique<CustomMid360DriverAdapter>();
        } else if (lidar_type == "unilidar") {
            lidar_adapter = std::make_unique<UnilidarAdapter>();
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "unknwon lidar type");
            rclcpp::shutdown();
            return;
        }
        lidar_adapter->setup_subscription(this, lidar_topic, [this](const std::vector<common::Point> &pointcloud) {
            small_point_lio->on_point_cloud_callback(pointcloud);
            small_point_lio->handle_once();
        });
        imu_subsciber = create_subscription<sensor_msgs::msg::Imu>(
                imu_topic,
                rclcpp::SensorDataQoS(),
                [this](const sensor_msgs::msg::Imu &msg) {
                    common::ImuMsg imu_msg;
                    imu_msg.angular_velocity = Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
                    imu_msg.linear_acceleration = Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
                    imu_msg.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
                    small_point_lio->on_imu_callback(imu_msg);
                    small_point_lio->handle_once();
                });
    }

    SmallPointLioNode::~SmallPointLioNode() {
        if (save_pcd_ && pointcloud_mapping) {
            RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "saving pcd on shutdown ...");
            auto points = pointcloud_mapping->get_points();
            if (pcd_tf_cached_) {
                for (auto &p : points) { p = R_pcd_ * p + T_pcd_; }
            }
            io::pcd::write_pcd(ROOT_DIR + "/pcd/scan.pcd", points);
            RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "save pcd success");
        }
    }

}// namespace small_point_lio

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(small_point_lio::SmallPointLioNode)

/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "small_point_lio.h"
#include <chrono>

namespace small_point_lio {

    using Clock = std::chrono::steady_clock;
    using ms_t  = std::chrono::duration<double, std::milli>;
    static double now_ms() {
        return ms_t(Clock::now().time_since_epoch()).count();
    }

    SmallPointLio::SmallPointLio(rclcpp::Node &node) {
        // init param
        parameters.read_parameters(node);
        preprocess.parameters = &parameters;
        estimator.parameters = &parameters;
        estimator.Lidar_T_wrt_IMU = parameters.extrinsic_T.cast<state::value_type>();
        estimator.Lidar_R_wrt_IMU = parameters.extrinsic_R.cast<state::value_type>();
        if (parameters.extrinsic_est_en) {
            estimator.kf.x.offset_T_L_I = parameters.extrinsic_T.cast<state::value_type>();
            estimator.kf.x.offset_R_L_I = parameters.extrinsic_R.cast<state::value_type>();
        }
        Q = estimator.process_noise_cov();
        estimator.imu_acceleration_scale = parameters.gravity.norm() / parameters.acc_norm;

        // init data
        reset();
    }

    void SmallPointLio::reset() {
        preprocess.reset();
        estimator.reset();
        is_init = false;
    }

    void SmallPointLio::on_point_cloud_callback(const std::vector<common::Point> &pointcloud) {
        preprocess.on_point_cloud_callback(pointcloud);
    }

    void SmallPointLio::on_imu_callback(const common::ImuMsg &imu_msg) {
        preprocess.on_imu_callback(imu_msg);
    }

    void SmallPointLio::handle_once() {
        // we need to init small point lio
        if (!is_init) {
            if ((!preprocess.point_deque.empty() || !preprocess.imu_deque.empty()) &&
                preprocess.point_deque.size() >= parameters.init_map_size &&
                (!parameters.fix_gravity_direction || preprocess.imu_deque.size() >= 200)) {
                // init map
                for (const auto &point: preprocess.point_deque) {
                    estimator.ivox->add_point(point.position);
                }
                // fix gravity direction
                if (parameters.fix_gravity_direction) {
                    estimator.kf.x.gravity = Eigen::Matrix<state::value_type, 3, 1>::Zero();
                    for (const auto &imu_msg: preprocess.imu_deque) {
                        estimator.kf.x.gravity += imu_msg.linear_acceleration.cast<state::value_type>();
                    }
                    state::value_type scale = -static_cast<state::value_type>(parameters.gravity.norm()) / estimator.kf.x.gravity.norm();
                    estimator.kf.x.gravity *= scale;
                } else {
                    estimator.kf.x.gravity = parameters.gravity.cast<state::value_type>();
                }
                estimator.kf.x.acceleration = -estimator.kf.x.gravity;
                // init time
                if (preprocess.point_deque.empty()) {
                    time_current = preprocess.imu_deque.back().timestamp;
                } else if (preprocess.imu_deque.empty()) {
                    time_current = preprocess.point_deque.back().timestamp;
                } else {
                    time_current = std::max(preprocess.point_deque.back().timestamp, preprocess.imu_deque.back().timestamp);
                }
                estimator.kf.init_timestamp(time_current);
                // clear data
                preprocess.point_deque.clear();
                preprocess.dense_point_deque.clear();
                preprocess.imu_deque.clear();
                is_init = true;
            }
            return;
        }

        // judge we should do point update or imu update
        bool is_publish_odometry = !preprocess.imu_deque.empty() && !preprocess.dense_point_deque.empty() && !preprocess.point_deque.empty() &&
                                   preprocess.imu_deque.front().timestamp < preprocess.point_deque.back().timestamp;
        auto _frame_t0 = Clock::now();
        while (!preprocess.imu_deque.empty() && !preprocess.dense_point_deque.empty() && !preprocess.point_deque.empty()) {
            const common::Point &point_lidar_frame = preprocess.point_deque.front();
            const common::Point &dense_point_lidar_frame = preprocess.dense_point_deque.front();
            const common::ImuMsg &imu_msg = preprocess.imu_deque.front();
            if (dense_point_lidar_frame.timestamp < point_lidar_frame.timestamp && dense_point_lidar_frame.timestamp < imu_msg.timestamp) {
                // collect lidar_odom frame pointcloud
                auto _t0 = Clock::now();
                Eigen::Matrix<state::value_type, 3, 1> dense_point_imu_frame;
                if (parameters.extrinsic_est_en) {
                    dense_point_imu_frame = estimator.kf.x.offset_R_L_I * dense_point_lidar_frame.position.cast<state::value_type>() + estimator.kf.x.offset_T_L_I;
                } else {
                    dense_point_imu_frame = estimator.Lidar_R_wrt_IMU * dense_point_lidar_frame.position.cast<state::value_type>() + estimator.Lidar_T_wrt_IMU;
                }
                pointcloud_odom_frame.emplace_back((estimator.kf.x.rotation * dense_point_imu_frame + estimator.kf.x.position).cast<float>());
                frame_dense_collect_.add(ms_t(Clock::now() - _t0).count());
                preprocess.dense_point_deque.pop_front();
            } else if (point_lidar_frame.timestamp < imu_msg.timestamp) {
                // point update
                if (point_lidar_frame.timestamp < time_current) {
                    double lag = time_current - point_lidar_frame.timestamp;
                    if (lag > 0.3) {
                        RCLCPP_WARN(rclcpp::get_logger("small_point_lio"),
                            "[point] stale point discarded: lag=%.3fs (timestamp=%.6f current=%.6f)",
                            lag, point_lidar_frame.timestamp, time_current);
                    }
                    preprocess.point_deque.pop_front();
                    continue;
                }
                double jump = point_lidar_frame.timestamp - time_current;
                if (jump > 0.3) {
                    RCLCPP_WARN(rclcpp::get_logger("small_point_lio"),
                        "[point] large timestamp jump: %.3fs (timestamp=%.6f current=%.6f)",
                        jump, point_lidar_frame.timestamp, time_current);
                }
                time_current = point_lidar_frame.timestamp;

                auto _pt0 = Clock::now();
                // predict
                estimator.kf.predict_state(time_current);

                // update
                estimator.point_lidar_frame = point_lidar_frame.position;
                estimator.kf.update_point();

                // publish odometry at fixed rate
                double publish_interval = 1.0 / parameters.odometry_publish_rate;
                if (last_odometry_publish_time < 0.0 || (time_current - last_odometry_publish_time) >= publish_interval) {
                    publish_odometry(time_current);
                    last_odometry_publish_time = time_current;
                }

                // map incremental
                estimator.ivox->add_point(estimator.point_odom_frame);
                frame_point_step_.add(ms_t(Clock::now() - _pt0).count());

                preprocess.point_deque.pop_front();
            } else {
                // imu update
                if (imu_msg.timestamp < time_current) {
                    double lag = time_current - imu_msg.timestamp;
                    if (lag > 0.3) {
                        RCLCPP_WARN(rclcpp::get_logger("small_point_lio"),
                            "[imu] stale imu discarded: lag=%.3fs (timestamp=%.6f current=%.6f)",
                            lag, imu_msg.timestamp, time_current);
                    }
                    preprocess.imu_deque.pop_front();
                    continue;
                }
                double jump = imu_msg.timestamp - time_current;
                if (jump > 0.3) {
                    RCLCPP_WARN(rclcpp::get_logger("small_point_lio"),
                        "[imu] large timestamp jump: %.3fs (timestamp=%.6f current=%.6f)",
                        jump, imu_msg.timestamp, time_current);
                }
                time_current = imu_msg.timestamp;

                auto _it0 = Clock::now();
                // predict
                estimator.kf.predict_state(time_current);
                estimator.kf.predict_cov(time_current, Q);

                // update
                estimator.angular_velocity = imu_msg.angular_velocity.cast<state::value_type>();
                estimator.linear_acceleration = imu_msg.linear_acceleration.cast<state::value_type>();
                estimator.kf.update_imu();
                frame_imu_step_.add(ms_t(Clock::now() - _it0).count());

                preprocess.imu_deque.pop_front();
            }
        }

        if (is_publish_odometry) {
            double compute_ms = ms_t(Clock::now() - _frame_t0).count();
            double now_wall = now_ms();
            double frame_interval = (last_frame_wall_time_ms_ < 0.0) ? compute_ms : (now_wall - last_frame_wall_time_ms_);
            last_frame_wall_time_ms_ = now_wall;

            const char *fmt =
                "[frame] compute=%.2fms interval=%.1fms | "
                "point_step: total=%.2fms n=%d | "
                "imu_step: total=%.2fms n=%d | "
                "dense_collect: total=%.2fms n=%d";

            if (compute_ms >= frame_interval) {
                RCLCPP_WARN(rclcpp::get_logger("small_point_lio"), fmt,
                    compute_ms, frame_interval,
                    frame_point_step_.total_ms,    frame_point_step_.count,
                    frame_imu_step_.total_ms,      frame_imu_step_.count,
                    frame_dense_collect_.total_ms, frame_dense_collect_.count);
            } else {
                bool should_log = (last_info_log_wall_time_ms_ < 0.0) ||
                                  (now_wall - last_info_log_wall_time_ms_ >= 1000.0);
                if (should_log) {
                    RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), fmt,
                        compute_ms, frame_interval,
                        frame_point_step_.total_ms,    frame_point_step_.count,
                        frame_imu_step_.total_ms,      frame_imu_step_.count,
                        frame_dense_collect_.total_ms, frame_dense_collect_.count);
                    last_info_log_wall_time_ms_ = now_wall;
                }
            }
            frame_point_step_.reset();
            frame_imu_step_.reset();
            frame_dense_collect_.reset();

            if (!pointcloud_odom_frame.empty()) {
                if (pointcloud_callback) {
                    pointcloud_callback(pointcloud_odom_frame);
                }
                pointcloud_odom_frame.clear();
            }
        }
    }

    void SmallPointLio::set_pointcloud_callback(const std::function<void(const std::vector<Eigen::Vector3f> &pointcloud)> &pointcloud_callback) {
        this->pointcloud_callback = pointcloud_callback;
    }

    void SmallPointLio::set_odometry_callback(const std::function<void(const common::Odometry &odometry)> &odometry_callback) {
        this->odometry_callback = odometry_callback;
    }

    void SmallPointLio::publish_odometry(double timestamp) {
        if (odometry_callback) {
            common::Odometry odometry;
            odometry.timestamp = timestamp;
            odometry.position = estimator.kf.x.position.cast<double>();
            odometry.velocity = estimator.kf.x.velocity.cast<double>();
            odometry.orientation = estimator.kf.x.rotation.cast<double>();
            odometry.angular_velocity = estimator.kf.x.omg.cast<double>();
            odometry_callback(odometry);
        }
    }

}// namespace small_point_lio

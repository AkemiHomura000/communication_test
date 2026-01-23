#!/usr/bin/env python3
import math
from collections import deque
from typing import Deque, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry

from robot_msg.msg import ChassisMsg as ChassisInfo


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time -> float seconds"""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class GyroDelayEstimator(Node):
    """
    Estimate time delay between:
      - LIO odom angular velocity (body frame): odom.twist.twist.angular.z
      - chassis angular velocity: chassis.wz + chassis.yaw_speed

    Using normalized cross-correlation over a sliding window.
    """

    def __init__(self):
        super().__init__('gyro_delay_estimator')

        # ---- Parameters ----
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('chassis_topic', '/chassis_info')

        self.declare_parameter('window_sec', 20.0)        # seconds used for correlation
        self.declare_parameter('lag_max_sec', 0.5)        # scan lag in [-lag_max, +lag_max]
        self.declare_parameter('lag_step_sec', 0.002)     # step size for scanning (2 ms)
        self.declare_parameter('resample_hz', 200.0)      # uniform resampling frequency
        self.declare_parameter('min_motion_radz', 0.15)   # gate: ignore near-zero segments
        self.declare_parameter('compute_period_sec', 1.0) # how often to compute delay

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.chassis_topic = self.get_parameter('chassis_topic').get_parameter_value().string_value

        self.window_sec = float(self.get_parameter('window_sec').value)
        self.lag_max_sec = float(self.get_parameter('lag_max_sec').value)
        self.lag_step_sec = float(self.get_parameter('lag_step_sec').value)
        self.resample_hz = float(self.get_parameter('resample_hz').value)
        self.min_motion_radz = float(self.get_parameter('min_motion_radz').value)
        self.compute_period_sec = float(self.get_parameter('compute_period_sec').value)

        # ---- Buffers: (t_sec, omega_z) ----
        self.lio_buf: Deque[Tuple[float, float]] = deque()
        self.odo_buf: Deque[Tuple[float, float]] = deque()

        # Subscribe
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 200)
        self.sub_chassis = self.create_subscription(ChassisInfo, self.chassis_topic, self.cb_chassis, 500)

        # Timer
        self.timer = self.create_timer(self.compute_period_sec, self.on_timer)

        self.get_logger().info(
            f"Listening odom: {self.odom_topic}, chassis: {self.chassis_topic}\n"
            f"window_sec={self.window_sec}, lag_max_sec=±{self.lag_max_sec}, "
            f"lag_step_sec={self.lag_step_sec}, resample_hz={self.resample_hz}"
        )

    def cb_odom(self, msg: Odometry):
        # Use header.stamp
        t = stamp_to_sec(msg.header.stamp)
        wz = float(msg.twist.twist.angular.z)  # body frame yaw rate
        self.lio_buf.append((t, wz))
        self._prune_buffers(now_t=t)

    def cb_chassis(self, msg: ChassisInfo):
        # Use builtin_interfaces/Time stamp field in chassis_info
        t = stamp_to_sec(msg.stamp)
        # Total yaw rate affecting sensor: base yaw rate + gimbal yaw rate
        wz_total = float(msg.wz) + float(msg.yaw_speed)
        self.odo_buf.append((t, wz_total))
        self._prune_buffers(now_t=t)

    def _prune_buffers(self, now_t: float):
        # keep enough history for correlation window + lag scan margin
        keep_sec = self.window_sec + self.lag_max_sec + 1.0
        t_min = now_t - keep_sec
        while self.lio_buf and self.lio_buf[0][0] < t_min:
            self.lio_buf.popleft()
        while self.odo_buf and self.odo_buf[0][0] < t_min:
            self.odo_buf.popleft()

    def on_timer(self):
        if len(self.lio_buf) < 50 or len(self.odo_buf) < 50:
            self.get_logger().warn("Not enough data yet.")
            return

        best_delay, best_corr = self.estimate_delay_ncc()
        if best_delay is None:
            self.get_logger().warn("Delay estimate failed (insufficient overlap or motion).")
            return

        # Interpretation:
        # We compute correlation between w_lio(t) and w_odo(t + delay).
        # If best_delay > 0: odo signal should be shifted forward to match lio (odo is earlier).
        self.get_logger().info(f"best_delay_sec={best_delay:+.4f}, best_corr={best_corr:.3f}")

    def estimate_delay_ncc(self):
        # Convert buffers to sorted arrays
        lio = np.array(self.lio_buf, dtype=np.float64)
        odo = np.array(self.odo_buf, dtype=np.float64)

        # Sort just in case
        lio = lio[np.argsort(lio[:, 0])]
        odo = odo[np.argsort(odo[:, 0])]

        t_lio, w_lio = lio[:, 0], lio[:, 1]
        t_odo, w_odo = odo[:, 0], odo[:, 1]

        # Define a common time grid based on overlap (without lag first)
        dt = 1.0 / self.resample_hz
        # Use the latest "window_sec" interval where both have data
        t_end = min(t_lio[-1], t_odo[-1])
        t_start = t_end - self.window_sec

        if t_start <= max(t_lio[0], t_odo[0]):
            # not enough overlap for a full window
            return None, None

        # Uniform grid
        t_grid = np.arange(t_start, t_end, dt)
        if t_grid.size < 100:
            return None, None

        # Interpolate LIO yaw rate on grid
        w_l = np.interp(t_grid, t_lio, w_lio)

        # Motion gate: ensure there is enough excitation
        if np.percentile(np.abs(w_l), 80) < self.min_motion_radz:
            return None, None

        # Prepare lag candidates
        lag_max = self.lag_max_sec
        lag_step = self.lag_step_sec
        lags = np.arange(-lag_max, lag_max + 1e-12, lag_step)

        # Normalize LIO sequence once
        wl = w_l.astype(np.float64)
        wl = wl - np.mean(wl)
        denom_wl = np.linalg.norm(wl)
        if denom_wl < 1e-9:
            return None, None

        best_corr = -1.0
        best_lag = None

        # For each lag, sample odo at shifted times (t + lag)
        # Correlate wl(t) with wo(t + lag)
        for lag in lags:
            # shifted time for odo
            t_shift = t_grid + lag

            # Require overlap inside odo interpolation bounds
            if t_shift[0] < t_odo[0] or t_shift[-1] > t_odo[-1]:
                continue

            wo = np.interp(t_shift, t_odo, w_odo).astype(np.float64)

            # Gate: also require motion in odo
            if np.percentile(np.abs(wo), 80) < self.min_motion_radz:
                continue

            wo = wo - np.mean(wo)
            denom_wo = np.linalg.norm(wo)
            if denom_wo < 1e-9:
                continue

            corr = float(np.dot(wl, wo) / (denom_wl * denom_wo))  # NCC in [-1,1]
            if corr > best_corr:
                best_corr = corr
                best_lag = float(lag)

        if best_lag is None:
            return None, None

        # Optional: try also negating odo to handle sign mismatch; uncomment if needed
        # (更严谨：你也可以在外面跑两次，选择相关性更高的符号)
        return best_lag, best_corr


def main():
    rclpy.init()
    node = GyroDelayEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
calib_lidar_yaw.py
==================
雷达 Yaw 外参（dpsi）标定工具。

原理
----
让机器人沿 **底盘 X 轴** 做纯平移（wz ≈ 0，vy_chassis ≈ 0），此时
底盘真实速度方向应与底盘 X 轴对齐。

若雷达安装存在 yaw 偏差 dpsi，LIO 里程计输出的 twist（机体帧）速度
实际上是在 "雷达帧" 下表示的，因此会观测到：
    vx_lio =  V * cos(dpsi)
    vy_lio =  V * sin(dpsi)
从而：
    dpsi = atan2(vy_lio, vx_lio)

节点做法
--------
1. 订阅 /Odometry (nav_msgs/Odometry)，取 twist.twist.linear.{x,y}
2. 每帧根据运动门限过滤（速度过低 / 角速度过大 视为无效）
3. 对每帧计算 dpsi_sample = atan2(vy, vx)，放入滑动窗口
4. 对窗口内样本求**循环均值**（circular mean），周期性发布结果

运行方法
--------
ros2 run lidar_extrinsic_calibration calib_lidar_yaw

可选参数（ros2 run ... --ros-args -p key:=value）：
  in_topic          默认 /Odometry
  window_sec        均值时间窗（秒），默认 10.0
  min_speed_mps     最小线速度门限（m/s），默认 0.15
  max_abs_omega     最大允许角速度（rad/s），默认 0.05
  min_samples       窗内最少有效样本，默认 20
  publish_rate_hz   发布频率，默认 10.0
  log_every_publish 每 N 次发布打印日志（0=不打印），默认 1
"""

import math
from collections import deque
from typing import Deque, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def circular_mean(angles: list) -> float:
    """
    Compute the circular (directional) mean of a list of angles (radians).
    Returns result in (-pi, pi].
    """
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum, cos_sum)


def circular_std(angles: list, mean_angle: float) -> float:
    """
    Circular standard deviation (radians).
    std = sqrt(-2 * ln(R)),  R = |mean unit vector|
    """
    n = len(angles)
    if n < 2:
        return 0.0
    R = math.sqrt(
        (sum(math.cos(a) for a in angles) / n) ** 2 +
        (sum(math.sin(a) for a in angles) / n) ** 2
    )
    R = min(R, 1.0 - 1e-12)  # 防止 log(0)
    return math.sqrt(-2.0 * math.log(R))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LidarYawCalibNode(Node):
    """
    Estimate LIO-to-chassis yaw extrinsic (dpsi) by observing
    the velocity direction in the LIO odometry frame while the
    robot moves purely along its X axis.
    """

    def __init__(self):
        super().__init__('lidar_yaw_calib')

        # ---- Parameters ----
        self.declare_parameter('in_topic',          '/Odometry')
        self.declare_parameter('window_sec',         10.0)
        self.declare_parameter('min_speed_mps',       0.15)   # 低速样本不可靠，丢弃
        self.declare_parameter('max_abs_omega',       0.05)   # 有转动则不是纯平移，丢弃
        self.declare_parameter('min_samples',          20)
        self.declare_parameter('publish_rate_hz',      10.0)
        self.declare_parameter('log_every_publish',     1)

        self.in_topic         = str(self.get_parameter('in_topic').value)
        self.window_sec       = float(self.get_parameter('window_sec').value)
        self.min_speed        = float(self.get_parameter('min_speed_mps').value)
        self.max_omega        = float(self.get_parameter('max_abs_omega').value)
        self.min_samples      = int(self.get_parameter('min_samples').value)
        self.publish_rate_hz  = float(self.get_parameter('publish_rate_hz').value)
        self.log_every        = int(self.get_parameter('log_every_publish').value)

        # buffer: (t_sec, dpsi_sample)
        self.buf: Deque[Tuple[float, float]] = deque()

        self.sub = self.create_subscription(
            Odometry, self.in_topic, self.cb_odom, 200
        )

        # Publish
        self.pub_dpsi     = self.create_publisher(Float64, '~/dpsi_deg',    10)
        self.pub_dpsi_rad = self.create_publisher(Float64, '~/dpsi_rad',    10)
        self.pub_std      = self.create_publisher(Float64, '~/dpsi_std_deg', 10)

        period = 1.0 / max(1e-6, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.on_timer)
        self._pub_count = 0

        self.get_logger().info(
            f"\n"
            f"  [LidarYawCalib] Subscribed: {self.in_topic}\n"
            f"  Publishing: ~/dpsi_deg, ~/dpsi_rad, ~/dpsi_std_deg\n"
            f"  window_sec={self.window_sec} s\n"
            f"  min_speed={self.min_speed} m/s  (只使用速度大于此值的样本)\n"
            f"  max_abs_omega={self.max_omega} rad/s  (角速度超过此值则丢弃)\n"
            f"  min_samples={self.min_samples}\n"
            f"\n"
            f"  操作方法：让机器人沿底盘 X 轴做纯直线平移，\n"
            f"  节点将自动统计 dpsi = atan2(vy, vx) 的均值。\n"
        )

    # ------------------------------------------------------------------
    def cb_odom(self, msg: Odometry):
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)

        speed = math.hypot(vx, vy)

        # 门限过滤：速度过低 或 正在转向 → 丢弃
        if speed < self.min_speed:
            return
        if abs(wz) > self.max_omega:
            return

        dpsi = math.atan2(vy, vx)   # 单位：rad，在 (-pi, pi]

        t = stamp_to_sec(msg.header.stamp)
        self.buf.append((t, dpsi))

        # 剪裁旧数据
        t_min = t - self.window_sec
        while self.buf and self.buf[0][0] < t_min:
            self.buf.popleft()

    # ------------------------------------------------------------------
    def on_timer(self):
        if len(self.buf) < self.min_samples:
            self.get_logger().warn(
                f"[LidarYawCalib] 有效样本不足 "
                f"({len(self.buf)} < {self.min_samples})，"
                f"请让机器人沿 X 轴平移。",
                throttle_duration_sec=5.0,
            )
            return

        angles = [d for _, d in self.buf]
        dpsi_mean = circular_mean(angles)
        dpsi_std  = circular_std(angles, dpsi_mean)

        # Publish
        m_deg = Float64(); m_deg.data = math.degrees(dpsi_mean)
        m_rad = Float64(); m_rad.data = dpsi_mean
        m_std = Float64(); m_std.data = math.degrees(dpsi_std)

        self.pub_dpsi.publish(m_deg)
        self.pub_dpsi_rad.publish(m_rad)
        self.pub_std.publish(m_std)

        self._pub_count += 1
        if self.log_every > 0 and (self._pub_count % self.log_every == 0):
            t0 = self.buf[0][0]
            t1 = self.buf[-1][0]
            self.get_logger().info(
                f"[win={t1 - t0:.1f}s  n={len(angles)}]  "
                f"dpsi = {math.degrees(dpsi_mean):+.4f} deg  "
                f"({dpsi_mean:+.6f} rad)  "
                f"std = {math.degrees(dpsi_std):.4f} deg"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = LidarYawCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出时打印最终结果，便于直接抄写到配置文件
        try:
            if len(node.buf) >= node.min_samples:
                angles = [d for _, d in node.buf]
                dpsi_final = circular_mean(angles)
                dpsi_std   = circular_std(angles, dpsi_final)
                node.get_logger().info(
                    f"\n"
                    f"  ========== 最终标定结果 ==========\n"
                    f"  dpsi (yaw 外参) = {math.degrees(dpsi_final):+.4f} deg\n"
                    f"                  = {dpsi_final:+.6f} rad\n"
                    f"  std             = {math.degrees(dpsi_std):.4f} deg\n"
                    f"  样本数          = {len(angles)}\n"
                    f"  ===================================\n"
                    f"  请将上述值填入 extrinsic_test.py 的 dpsi_gl 参数，\n"
                    f"  或 offline_extrinsic_calib_se2.py 的 --dpsi0 初始值。\n"
                )
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

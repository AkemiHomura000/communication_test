#!/usr/bin/env python3
import math
from collections import deque
from typing import Deque, Tuple, Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Quaternion -> yaw (Z axis), assuming standard ROS quaternion (x,y,z,w)."""
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomOffsetEstimator(Node):
    def __init__(self):
        super().__init__('odom_offset_estimator')

        # ---- parameters ----
        self.declare_parameter('in_topic', '/Odometry')
        self.declare_parameter('window_sec', 4.0)         # 平均时间窗（秒）
        self.declare_parameter('min_abs_omega', 1e-3)     # |wz|小于此值不用于估计
        self.declare_parameter('min_samples', 10)         # 窗内样本不足不发布
        self.declare_parameter('publish_rate_hz', 20.0)   # 定时发布平均结果
        self.declare_parameter('vel_in_world', True)      # 你说 vx,vy 是世界系；若其实是机体系则设 False
        self.declare_parameter('log_every_publish', 10)    # 0不打印；例如10=每10次发布打印

        self.in_topic = str(self.get_parameter('in_topic').value)
        self.window_sec = float(self.get_parameter('window_sec').value)
        self.min_abs_omega = float(self.get_parameter('min_abs_omega').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.vel_in_world = bool(self.get_parameter('vel_in_world').value)
        self.log_every_publish = int(self.get_parameter('log_every_publish').value)

        # buffer: (t_sec, x_body_est, y_body_est)
        self.buf: Deque[Tuple[float, float, float]] = deque()

        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb_odom, 50)

        self.pub_offset = self.create_publisher(Vector3Stamped, 'offset_xy', 10)
        self.pub_radius = self.create_publisher(Float64, 'radius', 10)

        period = 1.0 / max(1e-6, self.publish_rate_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.publish_count = 0
        self.last_yaw: Optional[float] = None

        self.get_logger().info(
            f"Subscribed: {self.in_topic}\n"
            f"Publishing: ~offset_xy (Vector3Stamped), ~radius (Float64)\n"
            f"window_sec={self.window_sec}, min_abs_omega={self.min_abs_omega}, "
            f"min_samples={self.min_samples}, publish_rate_hz={self.publish_rate_hz}, vel_in_world={self.vel_in_world}"
        )

    def cb_odom(self, msg: Odometry):
        # --- yaw from pose ---
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        self.last_yaw = yaw

        # --- velocities ---
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)  # body-frame, but z parallel => ok

        if abs(wz) < self.min_abs_omega:
            return

        # If twist linear velocity is NOT in world, rotate it into world using yaw
        if not self.vel_in_world:
            c = math.cos(yaw)
            s = math.sin(yaw)
            # v_world = Rz(yaw) * v_body
            vx_w = c * vx - s * vy
            vy_w = s * vx + c * vy
        else:
            vx_w, vy_w = vx, vy

        # --- estimate r_world from v_world = w x r_world ---
        # v_x = -w * r_y, v_y = w * r_x
        rx_w =  vy_w / wz
        ry_w = -vx_w / wz

        # --- convert to body frame: r_body = Rz(-yaw) * r_world ---
        c = math.cos(yaw)
        s = math.sin(yaw)
        # Rz(-yaw) = [[c, s],[-s, c]]
        rx_b = c * rx_w + s * ry_w
        ry_b = -s * rx_w + c * ry_w

        t = self.get_clock().now().nanoseconds * 1e-9
        self.buf.append((t, rx_b, ry_b))

        # prune old samples
        t_min = t - self.window_sec
        while self.buf and self.buf[0][0] < t_min:
            self.buf.popleft()

    def on_timer(self):
        if len(self.buf) < self.min_samples:
            return

        xs = [p[1] for p in self.buf]
        ys = [p[2] for p in self.buf]
        x_avg = sum(xs) / len(xs)
        y_avg = sum(ys) / len(ys)
        radius = math.hypot(x_avg, y_avg)

        out = Vector3Stamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "body"  # 这里表示输出是在机体系下；你也可以改成 imu_link/base_link
        out.vector.x = float(x_avg)
        out.vector.y = float(y_avg)
        out.vector.z = 0.0
        self.pub_offset.publish(out)

        rmsg = Float64()
        rmsg.data = float(radius)
        self.pub_radius.publish(rmsg)

        self.publish_count += 1
        if self.log_every_publish > 0 and (self.publish_count % self.log_every_publish == 0):
            t0 = self.buf[0][0]
            t1 = self.buf[-1][0]
            yaw = self.last_yaw if self.last_yaw is not None else float('nan')
            self.get_logger().info(
                f"[win={t1 - t0:.2f}s n={len(self.buf)}] "
                f"offset_body=({x_avg:+.4f},{y_avg:+.4f}) m, r={radius:.4f} m, yaw={yaw:+.3f} rad"
            )


def main():
    rclpy.init()
    node = OdomOffsetEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

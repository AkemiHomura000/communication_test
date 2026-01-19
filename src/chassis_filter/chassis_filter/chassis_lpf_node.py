#!/usr/bin/env python3
from typing import Optional
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from robot_msg.msg import ChassisMsg


class ChassisLowPassFilter(Node):
    """
    One-pole low-pass filter:
      y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    where alpha in (0, 1]. Smaller alpha => stronger smoothing.
    """

    def __init__(self):
        super().__init__('chassis_lpf_node')

        # ---- Parameters ----
        self.declare_parameter('input_topic', '/chassis_info')
        self.declare_parameter('output_topic', '/chassis_info_lpf')
        self.declare_parameter('alpha', 0.3)  # 0~1
        self.declare_parameter('copy_stamp', True)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.alpha = float(self.get_parameter('alpha').value)
        self.copy_stamp = bool(self.get_parameter('copy_stamp').value)

        if not (0.0 < self.alpha <= 1.0):
            self.get_logger().warn(f'alpha {self.alpha} out of range (0,1], set to 1.0')
            self.alpha = 1.0

        # ---- QoS ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(ChassisMsg, self.output_topic, qos)
        self.sub = self.create_subscription(ChassisMsg, self.input_topic, self.cb, qos)

        # Filter state
        self._inited = False
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._wz: float = 0.0
        self._yaw: float = 0.0

        # For yaw diff -> yaw_speed
        self._prev_yaw: float = 0.0
        self._prev_stamp_ns: int = 0
        self._yaw_speed: float = 0.0

        self.get_logger().info(
            f'Listening: {self.input_topic} -> Publishing: {self.output_topic}, alpha={self.alpha}'
        )

    def _lpf(self, prev: float, x: float) -> float:
        return prev + self.alpha * (x - prev)

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        """Wrap angle to [-pi, pi)."""
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def cb(self, msg: ChassisMsg):
        # ---- timestamp -> dt ----
        stamp_ns = int(msg.stamp.sec) * 1_000_000_000 + int(msg.stamp.nanosec)

        yaw_now = float(msg.yaw)

        if not self._inited:
            # First sample: initialize without smoothing jump
            self._vx = float(msg.vx)
            self._vy = float(msg.vy)
            self._wz = float(msg.wz)
            self._yaw = yaw_now

            self._prev_yaw = yaw_now
            self._prev_stamp_ns = stamp_ns
            self._yaw_speed = 0.0  # 首帧没有差分

            self._inited = True
        else:
            # low-pass for vx/vy/wz
            self._vx = self._lpf(self._vx, float(msg.vx))
            self._vy = self._lpf(self._vy, float(msg.vy))
            self._wz = self._lpf(self._wz, float(msg.wz))

            # yaw no lpf (keep raw), but compute yaw_speed by diff
            dt = (stamp_ns - self._prev_stamp_ns) * 1e-9  # seconds
            if dt > 1e-6:  # 防止 dt=0 或极小导致爆炸
                dyaw = self._wrap_to_pi(yaw_now - self._prev_yaw)  # unwrap diff
                self._yaw_speed = dyaw / dt
            else:
                # dt 不合法则保持上一帧 yaw_speed
                pass

            self._yaw = yaw_now
            self._prev_yaw = yaw_now
            self._prev_stamp_ns = stamp_ns

        out = ChassisMsg()
        if self.copy_stamp:
            out.stamp = msg.stamp
        else:
            out.stamp = self.get_clock().now().to_msg()

        out.vx = float(self._vx)
        out.vy = float(self._vy)
        out.wz = float(self._wz)
        out.yaw = float(self._yaw)

        # ---- fill yaw_speed ----
        out.yaw_speed = float(self._yaw_speed)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ChassisLowPassFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

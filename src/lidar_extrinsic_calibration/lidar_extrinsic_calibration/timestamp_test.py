#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from robot_msg.msg import ChassisMsg


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class TopicTimePrinter(Node):
    """
    Subscribe:
      - /Odometry (nav_msgs/Odometry)
      - /chassis_info (robot_msg/ChassisMsg)

    Print for each received message:
      - receive time (node clock now)
      - message stamp time
      - delta = receive - stamp
    """

    def __init__(self):
        super().__init__("topic_time_printer")

        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("chassis_topic", "/chassis_info")
        self.declare_parameter("print_every_n", 1)      # print every N messages
        self.declare_parameter("use_throttle_ms", 0)    # 0 means no throttle

        self.odom_topic = self.get_parameter("odom_topic").value
        self.chassis_topic = self.get_parameter("chassis_topic").value
        self.print_every_n = int(self.get_parameter("print_every_n").value)
        self.throttle_ms = int(self.get_parameter("use_throttle_ms").value)

        self._odom_cnt = 0
        self._ch_cnt = 0

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 200)
        self.sub_chassis = self.create_subscription(ChassisMsg, self.chassis_topic, self.cb_chassis, 500)

        self.get_logger().info(
            f"TopicTimePrinter started.\n"
            f"  odom_topic   : {self.odom_topic}\n"
            f"  chassis_topic: {self.chassis_topic}\n"
            f"  print_every_n: {self.print_every_n}\n"
            f"  throttle_ms  : {self.throttle_ms} (0=off)\n"
            f"Print fields: recv_time, msg_stamp, delta(recv-stamp)"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _should_print(self, cnt: int) -> bool:
        return (self.print_every_n <= 1) or (cnt % self.print_every_n == 0)

    def cb_odom(self, msg: Odometry):
        self._odom_cnt += 1
        if not self._should_print(self._odom_cnt):
            return

        t_recv = self._now_sec()
        t_stamp = stamp_to_sec(msg.header.stamp)
        dt = t_recv - t_stamp

        # Optional throttle (useful if you have very high rate)
        if self.throttle_ms > 0:
            self.get_logger().info_throttle(
                self.get_clock(), self.throttle_ms,
                f"[Odometry] recv={t_recv:.9f}  stamp={t_stamp:.9f}  dt={dt*1000.0:.3f} ms  frame={msg.header.frame_id}"
            )
        else:
            self.get_logger().info(
                f"[Odometry] recv={t_recv:.9f}  stamp={t_stamp:.9f}  dt={dt*1000.0:.3f} ms  frame={msg.header.frame_id}"
            )

    def cb_chassis(self, msg: ChassisMsg):
        self._ch_cnt += 1
        if not self._should_print(self._ch_cnt):
            return

        t_recv = self._now_sec()
        t_stamp = stamp_to_sec(msg.stamp)
        dt = t_recv - t_stamp

        if self.throttle_ms > 0:
            self.get_logger().info_throttle(
                self.get_clock(), self.throttle_ms,
                f"[Chassis]  recv={t_recv:.9f}  stamp={t_stamp:.9f}  dt={dt*1000.0:.3f} ms  "
                f"vx={float(msg.vx):+.3f} vy={float(msg.vy):+.3f} wz={float(msg.wz):+.3f}"
            )
        else:
            self.get_logger().info(
                f"[Chassis]  recv={t_recv:.9f}  stamp={t_stamp:.9f}  dt={dt*1000.0:.3f} ms  "
                f"vx={float(msg.vx):+.3f} vy={float(msg.vy):+.3f} wz={float(msg.wz):+.3f}"
            )


def main():
    rclpy.init()
    node = TopicTimePrinter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

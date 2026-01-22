#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)

    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=float)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]], dtype=float)

    return Rz @ Ry @ Rx


class ImuRollPitchAvgNode(Node):
    def __init__(self):
        super().__init__('imu_roll_pitch_avg_to_rotmat')

        self.topic = '/livox/imu_192_168_1_183'

        # --- parameters ---
        self.declare_parameter('avg_window', 50)     # 平均窗口长度（帧数）
        self.declare_parameter('min_samples', 10)    # 少于这个帧数不输出
        self.declare_parameter('log_every', 10)      # 每隔多少次回调打印一次

        self.avg_window = int(self.get_parameter('avg_window').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.log_every = int(self.get_parameter('log_every').value)

        self.buf = deque(maxlen=max(1, self.avg_window))
        self.count = 0

        self.sub = self.create_subscription(Imu, self.topic, self.cb_imu, 50)
        self.get_logger().info(
            f"Subscribed to: {self.topic}, avg_window={self.avg_window}, min_samples={self.min_samples}"
        )

    def cb_imu(self, msg: Imu):
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)

        # 1) push to buffer
        self.buf.append((ax, ay, az))
        n = len(self.buf)
        self.count += 1

        if n < self.min_samples:
            # 样本太少先不输出
            return

        # 2) average
        sx = sum(v[0] for v in self.buf)
        sy = sum(v[1] for v in self.buf)
        sz = sum(v[2] for v in self.buf)
        axm, aym, azm = sx / n, sy / n, sz / n

        # 3) normalize AFTER averaging (more stable)
        norm = math.sqrt(axm * axm + aym * aym + azm * azm)
        if norm < 1e-6:
            self.get_logger().warn("Averaged acceleration norm too small; skip.")
            return
        axm /= norm
        aym /= norm
        azm /= norm

        # 4) roll/pitch from accel (your convention: flat -> [0,0,+1], Iz up)
        roll = math.atan2(aym, azm)
        pitch = math.atan2(-axm, math.sqrt(aym * aym + azm * azm))
        yaw = 0.0

        R = rpy_to_rotmat(roll, pitch, yaw)

        # 5) log occasionally
        if self.count % max(1, self.log_every) == 0:
            self.get_logger().info(
                f"[n={n}] a_avg=({axm:+.4f},{aym:+.4f},{azm:+.4f}) "
                f"roll={roll/np.pi*180:+.4f} deg pitch={pitch/np.pi*180:+.4f} deg\n"
                f"R=\n{np.array2string(R, formatter={'float_kind':lambda x: f'{x: .6f}'})}"
            )
            yaw_chk = math.atan2(R[1,0], R[0,0])
            pitch_chk = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))
            roll_chk = math.atan2(R[2,1], R[2,2])
            self.get_logger().info(f"check yaw={yaw_chk:.6f}, pitch={pitch_chk:.6f}, roll={roll_chk:.6f}")



def main():
    rclpy.init()
    node = ImuRollPitchAvgNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

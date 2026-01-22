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


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """ZYX roll/pitch/yaw to quaternion (x,y,z,w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # q = qz * qy * qx
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr
    return (qx, qy, qz, qw)


class ImuLevelAndRepublishNode(Node):
    def __init__(self):
        super().__init__('imu_level_and_republish')

        # ---- parameters ----
        self.declare_parameter('in_topic', '/livox/imu_192_168_1_133')
        self.declare_parameter('out_topic', '/livox/imu_192_168_1_133_leveled')
        self.declare_parameter('avg_window', 100)      # 平均窗口帧数
        self.declare_parameter('min_samples', 10)     # 小于该帧数不发布
        self.declare_parameter('use_avg_for_output_accel', False)  # 输出加速度是否用平均值
        self.declare_parameter('set_orientation', True)            # 是否在输出IMU里填orientation(roll/pitch,yaw=0)
        self.declare_parameter('log_every', 20)        # 0=不打印；例如20=每20帧打印一次

        self.in_topic = str(self.get_parameter('in_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.avg_window = int(self.get_parameter('avg_window').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        self.use_avg_for_output_accel = bool(self.get_parameter('use_avg_for_output_accel').value)
        self.set_orientation = bool(self.get_parameter('set_orientation').value)
        self.log_every = int(self.get_parameter('log_every').value)

        self.buf = deque(maxlen=max(1, self.avg_window))
        self.count = 0

        self.sub = self.create_subscription(Imu, self.in_topic, self.cb_imu, 100)
        self.pub = self.create_publisher(Imu, self.out_topic, 100)

        self.get_logger().info(
            f"Subscribed: {self.in_topic}\n"
            f"Publishing: {self.out_topic}\n"
            f"avg_window={self.avg_window}, min_samples={self.min_samples}, "
            f"use_avg_for_output_accel={self.use_avg_for_output_accel}, set_orientation={self.set_orientation}"
        )

    def cb_imu(self, msg: Imu):
        # current raw vectors in I frame
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)

        gx = float(msg.angular_velocity.x)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)

        # 1) buffer accel for roll/pitch estimation
        self.buf.append((ax, ay, az))
        n = len(self.buf)
        self.count += 1
        if n < self.min_samples:
            return

        # 2) average accel -> normalize (for stable tilt estimation)
        axm = sum(v[0] for v in self.buf) / n
        aym = sum(v[1] for v in self.buf) / n
        azm = sum(v[2] for v in self.buf) / n

        norm = math.sqrt(axm * axm + aym * aym + azm * azm)
        if norm < 1e-6:
            self.get_logger().warn("Averaged accel norm too small; skip.")
            return
        axm /= norm
        aym /= norm
        azm /= norm

        # 3) roll/pitch from accel (your convention: static flat -> [0,0,+1], Iz up)
        roll = math.atan2(aym, azm)
        pitch = math.atan2(-axm, math.sqrt(aym * aym + azm * azm))
        yaw = 0.0

        # 4) Rotation matrix I -> L (leveled frame)
        R = rpy_to_rotmat(roll, pitch, yaw)

        # choose which accel to rotate & publish
        if self.use_avg_for_output_accel:
            aI = np.array([axm, aym, azm], dtype=float)
        else:
            aI = np.array([ax, ay, az], dtype=float)

        wI = np.array([gx, gy, gz], dtype=float)

        aL = R @ aI
        wL = R @ wI

        # 5) publish new IMU
        out = Imu()
        out.header = msg.header
        # 建议换一个frame_id，避免别人误以为还是原机体系数据
        out.header.frame_id = (msg.header.frame_id + "_leveled") if msg.header.frame_id else "imu_leveled"

        out.linear_acceleration.x = float(aL[0])
        out.linear_acceleration.y = float(aL[1])
        out.linear_acceleration.z = float(aL[2])

        out.angular_velocity.x = float(wL[0])
        out.angular_velocity.y = float(wL[1])
        out.angular_velocity.z = float(wL[2])

        # covariances: copy through (你也可以按需修改)
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        if self.set_orientation:
            qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
            out.orientation.x = float(qx)
            out.orientation.y = float(qy)
            out.orientation.z = float(qz)
            out.orientation.w = float(qw)
            out.orientation_covariance = msg.orientation_covariance
        else:
            # 不提供orientation（ROS约定：covariance[0] = -1 表示不可用）
            out.orientation_covariance[0] = -1.0

        self.pub.publish(out)

        # optional log
        if self.log_every > 0 and (self.count % self.log_every == 0):
            yaw_chk = math.atan2(R[1, 0], R[0, 0])
            self.get_logger().info(
                f"[n={n}] roll={roll/np.pi*180.0:+.4f}, pitch={pitch/np.pi*180.0:+.4f}, yaw_chk={yaw_chk/np.pi*180.0:+.4f}\n"
                f"aL=({aL[0]:+.4f},{aL[1]:+.4f},{aL[2]:+.4f}) "
                f"wL=({wL[0]:+.4f},{wL[1]:+.4f},{wL[2]:+.4f})"
            )


def main():
    rclpy.init()
    node = ImuLevelAndRepublishNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

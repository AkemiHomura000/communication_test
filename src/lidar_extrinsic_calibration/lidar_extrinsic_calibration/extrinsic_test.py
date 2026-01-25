#!/usr/bin/env python3
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from robot_msg.msg import ChassisMsg  # 你给的类型


# =========================
# Hard-coded fixed delay for /Odomtry
# =========================
# If /Odomtry has a fixed delay, compensate it here.
# Convention in this code:
#   t_odom_effective = t_odom_stamp + ODOM_DELAY_SEC
# Meaning ODOM_DELAY_SEC > 0 shifts odom timeline forward.
# If you find it should be the opposite, set ODOM_DELAY_SEC negative.
ODOM_DELAY_SEC = 0.01  # <-- 修改这里（单位：秒）


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Return 3x3 rotation matrix R such that: v_parent = R * v_child
    (Assuming quaternion represents child orientation in parent frame, as in nav_msgs/Odometry pose)
    """
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
        [2.0*(xy + wz),           1.0 - 2.0*(xx + zz),   2.0*(yz - wx)],
        [2.0*(xz - wy),           2.0*(yz + wx),         1.0 - 2.0*(xx + yy)]
    ], dtype=np.float64)
    return R


@dataclass
class OdomStamped:
    t_eff: float   # effective time after delay compensation
    msg: Odometry


class ChassisToMappedOdom(Node):
    """
    Subscribes:
      - /chassis_info (robot_msg/ChassisMsg): vx, vy, wz, yaw(phi), yaw_speed(phi_dot), stamp
      - /Odomtry (nav_msgs/Odometry): LIO pose in lidar_odom

    Computes:
      1) Convert chassis twist -> LIO frame twist (v_lio, w_lio) using extrinsics:
           p_bl(phi) = p_bg + Rz(phi) p_gl
           v_l^b = v_b + wz x p_bl + Rz(phi) (phi_dot x p_gl)
           R_bl = Rz(phi) * R_gl, with R_gl = Rz(dpsi)
           v_l^lio = R_lb * v_l^b
           w_l^lio = [0,0,wz + phi_dot] (planar)
      2) Rotate v_l^lio into lidar_odom using odom orientation:
           v_lidar_odom = R_lidar_odom_lio * v_lio

    Publishes:
      - /wheel_mapped_odom (nav_msgs/Odometry):
           pose copied from matched /Odomtry
           twist.linear = v in lidar_odom
           twist.angular = w in lio  (mixed-frame semantics as requested)
    """

    def __init__(self):
        super().__init__("chassis_to_mapped_odom")

        # -------- Params (extrinsics) --------
        self.declare_parameter("p_gl_x", -0.082)   # meters, g->l in g frame
        self.declare_parameter("p_gl_y", 0.168)
        self.declare_parameter("p_bg_x", 0.0)      # meters, b->g in b frame
        self.declare_parameter("p_bg_y", 0.0)
        self.declare_parameter("dpsi_gl", 0.0)     # radians, yaw offset for R_gl = Rz(dpsi)

        # -------- Params (topics) --------
        self.declare_parameter("chassis_topic", "/chassis_info")
        self.declare_parameter("odom_topic", "/Odomtry")
        self.declare_parameter("out_topic", "/wheel_mapped_odom")

        # -------- Params (sync/buffer) --------
        self.declare_parameter("odom_buffer_sec", 2.0)      # how long odom msgs kept
        self.declare_parameter("max_sync_dt_sec", 0.05)     # max time diff to match (after delay compensation)

        self.p_gl = np.array([
            float(self.get_parameter("p_gl_x").value),
            float(self.get_parameter("p_gl_y").value),
        ], dtype=np.float64)

        self.p_bg = np.array([
            float(self.get_parameter("p_bg_x").value),
            float(self.get_parameter("p_bg_y").value),
        ], dtype=np.float64)

        self.dpsi = float(self.get_parameter("dpsi_gl").value)

        self.chassis_topic = self.get_parameter("chassis_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.out_topic = self.get_parameter("out_topic").value

        self.odom_buffer_sec = float(self.get_parameter("odom_buffer_sec").value)
        self.max_sync_dt = float(self.get_parameter("max_sync_dt_sec").value)

        # -------- Buffers --------
        self.odom_buf: Deque[OdomStamped] = deque()

        # -------- Subs/Pubs --------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 200)
        self.sub_chassis = self.create_subscription(ChassisMsg, self.chassis_topic, self.cb_chassis, 500)
        self.pub_out = self.create_publisher(Odometry, self.out_topic, 50)

        self.get_logger().info(
            f"Started. chassis={self.chassis_topic}, odom={self.odom_topic}, out={self.out_topic}\n"
            f"Extrinsics: p_gl=[{self.p_gl[0]:+.3f},{self.p_gl[1]:+.3f}] m, "
            f"p_bg=[{self.p_bg[0]:+.3f},{self.p_bg[1]:+.3f}] m, dpsi={self.dpsi:+.4f} rad\n"
            f"Hard-coded /Odomtry delay compensation: ODOM_DELAY_SEC={ODOM_DELAY_SEC:+.3f} s "
            f"(t_eff = stamp + delay)"
        )

    def cb_odom(self, msg: Odometry):
        t_stamp = stamp_to_sec(msg.header.stamp)
        # apply fixed delay compensation
        t_eff = t_stamp + ODOM_DELAY_SEC
        self.odom_buf.append(OdomStamped(t_eff=t_eff, msg=msg))
        self._prune_odom_buffer(now_t_eff=t_eff)

    def _prune_odom_buffer(self, now_t_eff: float):
        t_min = now_t_eff - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0].t_eff < t_min:
            self.odom_buf.popleft()

    def _find_nearest_odom(self, t_query: float) -> Optional[Odometry]:
        """
        Find odom with nearest effective time t_eff to t_query.
        """
        if not self.odom_buf:
            return None
        best_msg = None
        best_dt = 1e9
        for item in self.odom_buf:
            dt = abs(item.t_eff - t_query)
            if dt < best_dt:
                best_dt = dt
                best_msg = item.msg
        if best_msg is None or best_dt > self.max_sync_dt:
            return None
        return best_msg

    def cb_chassis(self, msg: ChassisMsg):
        # Update extrinsics each callback in case you change params dynamically
        self.p_gl[0] = float(self.get_parameter("p_gl_x").value)
        self.p_gl[1] = float(self.get_parameter("p_gl_y").value)
        self.p_bg[0] = float(self.get_parameter("p_bg_x").value)
        self.p_bg[1] = float(self.get_parameter("p_bg_y").value)
        self.dpsi = float(self.get_parameter("dpsi_gl").value)

        t_ch = stamp_to_sec(msg.stamp)

        # Find odom matched in "effective time axis"
        odom = self._find_nearest_odom(t_ch)
        if odom is None:
            self.get_logger().warn("No matched odom (time sync) for chassis sample.")
            return

        # -------- 1) chassis -> LIO frame twist --------
        vx = float(msg.vx)
        vy = float(msg.vy)
        wz = float(msg.wz)
        phi = float(msg.yaw)          # gimbal yaw relative base
        phi_dot = float(msg.yaw_speed)

        v_lio, w_lio = self._chassis_to_lio_twist(vx, vy, wz, phi, phi_dot)

        # -------- 2) LIO linear vel -> lidar_odom frame using odom orientation --------
        q = odom.pose.pose.orientation
        R_lidarodom_lio = quat_to_rotmat(float(q.x), float(q.y), float(q.z), float(q.w))
        v_lidarodom = R_lidarodom_lio @ v_lio

        # -------- 3) publish new Odometry --------
        out = Odometry()
        out.header = odom.header
        out.child_frame_id = odom.child_frame_id

        # Copy pose from LIO odom (lio pose in lidar_odom)
        out.pose = odom.pose

        # Twist:
        # - linear: expressed in lidar_odom frame (as you requested)
        # - angular: expressed in lio frame (as you requested; mixed-frame semantics)
        out.twist.twist.linear.x = float(v_lidarodom[0])
        out.twist.twist.linear.y = float(v_lidarodom[1])
        out.twist.twist.linear.z = float(v_lidarodom[2])

        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = float(w_lio[2])

        out.twist.covariance = odom.twist.covariance
        out.pose.covariance = odom.pose.covariance

        self.pub_out.publish(out)

    def _chassis_to_lio_twist(
        self,
        vx: float,
        vy: float,
        wz: float,
        phi: float,
        phi_dot: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          v_lio: 3x1 linear velocity expressed in LIO frame (assume planar z=0)
          w_lio: 3x1 angular velocity expressed in LIO frame (assume only z)
        Model (2D):
          p_bl(phi) = p_bg + Rz(phi) p_gl
          v_l^b = v_b + wz x p_bl + Rz(phi) (phi_dot x p_gl)
          R_bl = Rz(phi) * Rz(dpsi) = Rz(phi + dpsi)
          v_l^lio = R_lb * v_l^b  (R_lb = Rz(-(phi+dpsi)))
          w_l^lio = [0,0,wz + phi_dot]
        """
        p_gl = self.p_gl.copy()
        p_bg = self.p_bg.copy()

        Rbg = rot2(phi)  # rotate g-frame vectors into b-frame
        p_bl = p_bg + Rbg @ p_gl  # lidar position relative base in b frame

        v_b = np.array([vx, vy], dtype=np.float64)

        # wz x p_bl => [-w*y, w*x]
        v_w_cross = np.array([-wz * p_bl[1], wz * p_bl[0]], dtype=np.float64)

        # gimbal term: (phi_dot x p_gl) in g frame, rotate to b by Rbg
        v_g_cross_g = np.array([-phi_dot * p_gl[1], phi_dot * p_gl[0]], dtype=np.float64)
        v_g_cross_b = Rbg @ v_g_cross_g

        v_l_b = v_b + v_w_cross + v_g_cross_b

        # b -> lio rotation: R_bl = Rz(phi + dpsi), so R_lb = Rz(-(phi+dpsi))
        Rlb = rot2(-(phi + self.dpsi))
        v_l_lio_2d = Rlb @ v_l_b

        v_lio = np.array([v_l_lio_2d[0], v_l_lio_2d[1], 0.0], dtype=np.float64)
        w_lio = np.array([0.0, 0.0, wz + phi_dot], dtype=np.float64)
        return v_lio, w_lio


def main():
    rclpy.init()
    node = ChassisToMappedOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

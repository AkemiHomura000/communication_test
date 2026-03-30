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
# Optional fixed delay compensation on odom stamp axis
# =========================
# Convention:
#   t_odom_effective = t_odom_stamp + ODOM_DELAY_SEC
# Then we match chassis where: t_chassis_stamp >= t_odom_effective
# If you don't need it, set to 0.0
ODOM_DELAY_SEC = 0.40  # <-- 需要补偿再改


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Return 3x3 rotation matrix R such that: v_parent = R * v_child
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
class ChassisStamped:
    t: float
    msg: ChassisMsg


@dataclass
class OdomStamped:
    t: float
    msg: Odometry


class ChassisToMappedOdom(Node):
    """
    New logic:
      - Buffer chassis_info by its msg.stamp
      - Buffer odom by its header.stamp
      - When odom arrives (or chassis arrives), try to match:
          choose chassis whose stamp is AHEAD of odom_effective_time:
              t_chassis >= (t_odom + ODOM_DELAY_SEC)
          and with minimal positive lead (t_chassis - t_odom_eff)
      - Publish output with header.stamp EXACTLY equals odom.header.stamp
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
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("out_topic", "/wheel_mapped_odom")

        # -------- Params (buffers / matching) --------
        self.declare_parameter("chassis_buffer_sec", 2.0)
        self.declare_parameter("odom_buffer_sec", 2.0)

        # max allowed "lead": chassis ahead of odom by how much (seconds)
        # lead = t_chassis - t_odom_eff
        self.declare_parameter("max_lead_dt_sec", 0.07)

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

        self.chassis_buffer_sec = float(self.get_parameter("chassis_buffer_sec").value)
        self.odom_buffer_sec = float(self.get_parameter("odom_buffer_sec").value)
        self.max_lead_dt = float(self.get_parameter("max_lead_dt_sec").value)

        # -------- Buffers --------
        self.chassis_buf: Deque[ChassisStamped] = deque()
        self.odom_buf: Deque[OdomStamped] = deque()

        # -------- Subs/Pubs --------
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, 200)
        self.sub_chassis = self.create_subscription(ChassisMsg, self.chassis_topic, self.cb_chassis, 500)
        self.pub_out = self.create_publisher(Odometry, self.out_topic, 50)

        self.get_logger().info(
            f"Started.\n"
            f"  chassis={self.chassis_topic}\n"
            f"  odom   ={self.odom_topic}\n"
            f"  out    ={self.out_topic}\n"
            f"Extrinsics: p_gl=[{self.p_gl[0]:+.3f},{self.p_gl[1]:+.3f}] m, "
            f"p_bg=[{self.p_bg[0]:+.3f},{self.p_bg[1]:+.3f}] m, dpsi={self.dpsi:+.4f} rad\n"
            f"Matching rule: choose chassis with stamp >= (odom_stamp + ODOM_DELAY_SEC), "
            f"minimize lead, lead<=max_lead_dt.\n"
            f"ODOM_DELAY_SEC={ODOM_DELAY_SEC:+.3f}, max_lead_dt={self.max_lead_dt:.3f}"
        )

    # ----------------------------
    # Callbacks & Buffering
    # ----------------------------
    def cb_chassis(self, msg: ChassisMsg):
        # update extrinsics if you change params dynamically
        self.p_gl[0] = float(self.get_parameter("p_gl_x").value)
        self.p_gl[1] = float(self.get_parameter("p_gl_y").value)
        self.p_bg[0] = float(self.get_parameter("p_bg_x").value)
        self.p_bg[1] = float(self.get_parameter("p_bg_y").value)
        self.dpsi = float(self.get_parameter("dpsi_gl").value)

        t = stamp_to_sec(msg.stamp)
        self.chassis_buf.append(ChassisStamped(t=t, msg=msg))
        self._prune_chassis_buffer(now_t=t)

        # chassis 到来后也尝试处理 pending odom（保证不丢）
        self._try_process()

    def cb_odom(self, msg: Odometry):
        t = stamp_to_sec(msg.header.stamp)
        self.odom_buf.append(OdomStamped(t=t, msg=msg))
        self._prune_odom_buffer(now_t=t)

        # 每次 odom 到来都尝试匹配并发布
        self._try_process()

    def _prune_chassis_buffer(self, now_t: float):
        t_min = now_t - self.chassis_buffer_sec
        while self.chassis_buf and self.chassis_buf[0].t < t_min:
            self.chassis_buf.popleft()

    def _prune_odom_buffer(self, now_t: float):
        t_min = now_t - self.odom_buffer_sec
        while self.odom_buf and self.odom_buf[0].t < t_min:
            self.odom_buf.popleft()

    # ----------------------------
    # Matching & Publishing
    # ----------------------------
    def _find_first_ahead_chassis(self, t_odom_eff: float) -> Optional[Tuple[ChassisStamped, float]]:
        """
        Find chassis with timestamp >= t_odom_eff, and minimal lead.
        Return (chassis_item, lead_dt).
        """
        if not self.chassis_buf:
            return None

        best: Optional[ChassisStamped] = None
        best_lead = 1e9

        # chassis stamp must be ahead (>=)
        for item in self.chassis_buf:
            lead = item.t - t_odom_eff
            if lead >= 0.0 and lead < best_lead:
                best_lead = lead
                best = item

        if best is None:
            return None
        return best, best_lead

    def _try_process(self):
        """
        Process odom in time order:
          for the oldest odom, if we can find an "ahead" chassis, publish once, then pop that odom.
          repeat as long as possible.
        """
        while self.odom_buf:
            odom_item = self.odom_buf[0]
            t_odom = odom_item.t
            t_odom_eff = t_odom + ODOM_DELAY_SEC

            found = self._find_first_ahead_chassis(t_odom_eff)
            if found is None:
                # 还没等到时间戳超前的 chassis，先不弹出 odom，等待后续 chassis
                return

            ch_item, lead_dt = found
            if lead_dt > self.max_lead_dt:
                # 超前太多：这条 odom 很可能已经“错过”合适的 chassis
                # 策略：丢弃这条 odom，避免阻塞后续
                self.get_logger().warn(
                    f"Drop odom t={t_odom:.6f}: best ahead chassis lead_dt={lead_dt:.6f}s > max_lead_dt={self.max_lead_dt:.6f}s"
                )
                self.odom_buf.popleft()
                continue

            # 有匹配：发布
            self._publish_from_pair(odom_item.msg, ch_item.msg, lead_dt, t_odom_eff)

            # 发布后弹出这条 odom（每条 odom 只发布一次）
            self.odom_buf.popleft()

            # 可选：为了避免 chassis 被重复使用（如果你希望“一条 chassis 只用一次”）
            # 这里可以把时间 <= ch_item.t 的 chassis 都弹掉，或只弹掉该条
            # 我这里采取“弹掉早于等于已使用 chassis 的所有旧 chassis”，避免反复命中同一条
            while self.chassis_buf and self.chassis_buf[0].t <= ch_item.t:
                self.chassis_buf.popleft()

    def _publish_from_pair(self, odom: Odometry, chassis: ChassisMsg, lead_dt: float, t_odom_eff: float):
        # -------- 1) chassis -> LIO frame twist --------
        vx = float(chassis.vx)
        vy = float(chassis.vy)
        wz = float(chassis.wz)
        phi = float(chassis.yaw)
        phi_dot = float(chassis.yaw_speed)

        v_lio, w_lio = self._chassis_to_lio_twist(vx, vy, wz, phi, phi_dot)

        # -------- 2) LIO linear vel -> lidar_odom frame using odom orientation --------
        q = odom.pose.pose.orientation
        R_lidarodom_lio = quat_to_rotmat(float(q.x), float(q.y), float(q.z), float(q.w))
        v_lidarodom = R_lidarodom_lio @ v_lio

        # -------- 3) publish new Odometry --------
        out = Odometry()
        out.header = odom.header
        # 关键：保证输出时间戳与 /Odometry 重合
        out.header.stamp = odom.header.stamp

        out.child_frame_id = odom.child_frame_id
        out.pose = odom.pose

        out.twist.twist.linear.x = float(v_lidarodom[0])
        out.twist.twist.linear.y = float(v_lidarodom[1])
        out.twist.twist.linear.z = float(v_lidarodom[2])

        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = float(w_lio[2])

        out.twist.covariance = odom.twist.covariance
        out.pose.covariance = odom.pose.covariance

        self.pub_out.publish(out)

        # 可选：打印匹配情况（节流避免刷屏）
        self.get_logger().debug(
            f"match ok: odom_stamp={stamp_to_sec(odom.header.stamp):.6f} "
            f"odom_eff={t_odom_eff:.6f} chassis_stamp={stamp_to_sec(chassis.stamp):.6f} "
            f"lead_dt={lead_dt*1000.0:.3f} ms"
        )

    # ----------------------------
    # Kinematics conversion
    # ----------------------------
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
          v_lio: 3x1 linear velocity in LIO frame
          w_lio: 3x1 angular velocity in LIO frame (only z)
        Model (2D):
          p_bl(phi) = p_bg + Rz(phi) p_gl
          v_l^b = v_b + wz x p_bl + Rz(phi) (phi_dot x p_gl)
          R_bl = Rz(phi + dpsi), so R_lb = Rz(-(phi+dpsi))
          v_l^lio = R_lb * v_l^b
          w_l^lio = [0,0,wz + phi_dot]
        """
        p_gl = self.p_gl.copy()
        p_bg = self.p_bg.copy()

        Rbg = rot2(phi)
        p_bl = p_bg + Rbg @ p_gl

        v_b = np.array([vx, vy], dtype=np.float64)

        v_w_cross = np.array([-wz * p_bl[1], wz * p_bl[0]], dtype=np.float64)

        v_g_cross_g = np.array([-phi_dot * p_gl[1], phi_dot * p_gl[0]], dtype=np.float64)
        v_g_cross_b = Rbg @ v_g_cross_g

        v_l_b = v_b + v_w_cross + v_g_cross_b

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

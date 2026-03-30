#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ROS2 bag reading
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from nav_msgs.msg import Odometry
from robot_msg.msg import ChassisMsg  # 你给的类型

# Optimization
from scipy.optimize import least_squares


# ----------------------------
# Math helpers
# ----------------------------
def wrap_to_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # yaw from quaternion (Z axis)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class OdomSample:
    t: float
    x: float
    y: float
    yaw: float


@dataclass
class ChassisSample:
    t: float
    vx: float
    vy: float
    wz: float
    phi: float         # yaw (gimbal relative base)
    phi_dot: float     # yaw_speed


# ----------------------------
# Bag reading
# ----------------------------
def read_bag(bag_path: str, odom_topic: str, chassis_topic: str) -> Tuple[List[OdomSample], List[ChassisSample]]:
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics_and_types}

    if odom_topic not in type_map:
        raise RuntimeError(f"odom_topic '{odom_topic}' not found in bag. Available: {list(type_map.keys())}")
    if chassis_topic not in type_map:
        raise RuntimeError(f"chassis_topic '{chassis_topic}' not found in bag. Available: {list(type_map.keys())}")

    odom_type = get_message(type_map[odom_topic])
    chassis_type = get_message(type_map[chassis_topic])

    odoms: List[OdomSample] = []
    chass: List[ChassisSample] = []

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == odom_topic:
            msg = deserialize_message(data, odom_type)
            # Expect nav_msgs/Odometry
            if not isinstance(msg, Odometry):
                # but in case get_message returns a dynamic class, still treat as Odometry-like
                pass
            t = stamp_to_sec(msg.header.stamp)
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
            odoms.append(OdomSample(t=t, x=x, y=y, yaw=yaw))

        elif topic == chassis_topic:
            msg = deserialize_message(data, chassis_type)
            # Expect robot_msg/ChassisMsg with fields:
            # builtin_interfaces/Time stamp
            # float32 vx, vy, wz, yaw, yaw_speed
            t = stamp_to_sec(msg.stamp)
            chass.append(ChassisSample(
                t=t,
                vx=float(msg.vx),
                vy=float(msg.vy),
                wz=float(msg.wz),
                phi=float(msg.yaw),
                phi_dot=float(msg.yaw_speed),
            ))

    odoms.sort(key=lambda s: s.t)
    chass.sort(key=lambda s: s.t)

    if len(odoms) < 2:
        raise RuntimeError("Not enough odom samples.")
    if len(chass) < 2:
        raise RuntimeError("Not enough chassis samples.")

    return odoms, chass


# ----------------------------
# Interpolation utilities for chassis signals
# ----------------------------
class ChassisInterp:
    def __init__(self, samples: List[ChassisSample]):
        self.t = np.array([s.t for s in samples], dtype=np.float64)
        self.vx = np.array([s.vx for s in samples], dtype=np.float64)
        self.vy = np.array([s.vy for s in samples], dtype=np.float64)
        self.wz = np.array([s.wz for s in samples], dtype=np.float64)
        self.phi = np.array([s.phi for s in samples], dtype=np.float64)
        self.phi_dot = np.array([s.phi_dot for s in samples], dtype=np.float64)

        # unwrap phi to avoid discontinuity around +/-pi
        self.phi = np.unwrap(self.phi)

    def in_range(self, t0: float, t1: float) -> bool:
        return (t0 >= self.t[0]) and (t1 <= self.t[-1])

    def sample(self, tq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vx = np.interp(tq, self.t, self.vx)
        vy = np.interp(tq, self.t, self.vy)
        wz = np.interp(tq, self.t, self.wz)
        phi = np.interp(tq, self.t, self.phi)
        phi_dot = np.interp(tq, self.t, self.phi_dot)
        return vx, vy, wz, phi, phi_dot


# ----------------------------
# Odo-predicted relative increment integration (SE2)
# Params: [pgl_x, pgl_y, pbg_x, pbg_y, delta_psi]
# ----------------------------
def integrate_predicted_delta(
    interp: ChassisInterp,
    t0: float,
    t1: float,
    params: np.ndarray,
    step_dt: float = 0.01
) -> Optional[Tuple[float, float, float]]:
    """
    Predict relative motion of lidar between t0 and t1 in lidar frame at t0:
      returns (dx, dy, dyaw) in SE(2) relative transform.

    Uses chassis signals: vx,vy,wz,phi,phi_dot
    Model:
      p_bl(phi) = p_bg + Rz(phi) p_gl
      v_l^b = v_b + wz x p_bl + Rz(phi) (phi_dot x p_gl)
      R_bl = Rz(phi) * Rz(delta_psi) = Rz(phi + delta_psi)
      v_l^l = R_lb * v_l^b
      yaw_rate_l = wz + phi_dot
    Then integrate SE(2) in start-lidar frame.
    """
    pgl_x, pgl_y, pbg_x, pbg_y, dpsi = [float(x) for x in params]
    p_gl = np.array([pgl_x, pgl_y], dtype=np.float64)
    p_bg = np.array([pbg_x, pbg_y], dtype=np.float64)

    if t1 <= t0:
        return None
    if not interp.in_range(t0, t1):
        return None

    # time grid
    n = int(math.ceil((t1 - t0) / step_dt))
    if n < 1:
        n = 1
    tq = np.linspace(t0, t1, n + 1, dtype=np.float64)
    dt = np.diff(tq)

    vx, vy, wz, phi, phi_dot = interp.sample(tq[:-1])

    # SE2 integration state (relative pose from t0 to current, expressed in t0 lidar frame)
    dx = 0.0
    dy = 0.0
    dtheta = 0.0  # relative yaw from t0 lidar to current lidar

    for i in range(len(dt)):
        dti = float(dt[i])
        w = float(wz[i])
        ph = float(phi[i])
        phdot = float(phi_dot[i])

        # p_bl in b frame (2D)
        Rbg = rot2(ph)
        p_bl = p_bg + Rbg @ p_gl

        # v_b in b frame
        v_b = np.array([float(vx[i]), float(vy[i])], dtype=np.float64)

        # wz x p_bl (2D): [-w*y, w*x]
        v_w_cross = np.array([-w * p_bl[1], w * p_bl[0]], dtype=np.float64)

        # gimbal rotation term: phi_dot x p_gl in g frame, then rotate to b by Rbg
        v_g_cross_g = np.array([-phdot * p_gl[1], phdot * p_gl[0]], dtype=np.float64)
        v_g_cross_b = Rbg @ v_g_cross_g

        v_l_b = v_b + v_w_cross + v_g_cross_b

        # Rotate b->l: R_bl = Rz(phi + dpsi) so R_lb = Rz(-(phi + dpsi))
        Rlb = rot2(-(ph + dpsi))
        v_l_l = Rlb @ v_l_b

        # yaw rate of lidar (around z)
        w_l = w + phdot

        # integrate: position in start frame += R(dtheta) * v_body * dt
        Rstart_to_curr = rot2(dtheta)
        v_in_start = Rstart_to_curr @ v_l_l
        dx += float(v_in_start[0] * dti)
        dy += float(v_in_start[1] * dti)
        dtheta += float(w_l * dti)

    return dx, dy, wrap_to_pi(dtheta)


# ----------------------------
# LIO relative increment from odom samples
# ----------------------------
def odom_relative_increment(o0: OdomSample, o1: OdomSample) -> Tuple[float, float, float]:
    dyaw = wrap_to_pi(o1.yaw - o0.yaw)
    dp_w = np.array([o1.x - o0.x, o1.y - o0.y], dtype=np.float64)
    # express translation in frame of o0 (lidar body at t0): R(-yaw0) * dp_world
    dp_0 = rot2(-o0.yaw) @ dp_w
    return float(dp_0[0]), float(dp_0[1]), float(dyaw)


# ----------------------------
# Residual construction
# ----------------------------
def build_residuals(
    params: np.ndarray,
    odoms: List[OdomSample],
    interp: ChassisInterp,
    step_dt: float,
    min_dt: float,
    max_dt: float,
    motion_gate_w: float,
    motion_gate_v: float,
    prior: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Stack residuals for all usable consecutive odom pairs.
    prior: (mean, std) for params, same shape as params (use std=inf to disable each term)
    """
    res = []
    for k in range(len(odoms) - 1):
        o0, o1 = odoms[k], odoms[k + 1]
        dt = o1.t - o0.t
        if dt < min_dt or dt > max_dt:
            continue

        # LIO relative
        dx_lio, dy_lio, dyaw_lio = odom_relative_increment(o0, o1)

        # Quick motion gate using LIO increments (robust against chassis noise)
        if abs(dyaw_lio) < motion_gate_w and math.hypot(dx_lio, dy_lio) < motion_gate_v:
            continue

        pred = integrate_predicted_delta(interp, o0.t, o1.t, params, step_dt=step_dt)
        if pred is None:
            continue
        dx_odo, dy_odo, dyaw_odo = pred

        # residual (SE2)
        res.append(dx_lio - dx_odo)
        res.append(dy_lio - dy_odo)
        res.append(wrap_to_pi(dyaw_lio - dyaw_odo))

    if len(res) == 0:
        # return something non-empty to avoid optimizer crash
        res = [0.0, 0.0, 0.0]

    # priors
    if prior is not None:
        mean, std = prior
        for i in range(len(params)):
            si = float(std[i])
            if not math.isfinite(si) or si <= 0:
                continue
            res.append((float(params[i]) - float(mean[i])) / si)

    return np.array(res, dtype=np.float64)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to rosbag2 folder (contains metadata.yaml)")
    ap.add_argument("--odom_topic", default="/Odomtry", help="Odometry topic, default /Odomtry")
    ap.add_argument("--chassis_topic", default="/chassis_info", help="Chassis topic name in bag")
    ap.add_argument("--step_dt", type=float, default=0.01, help="Integration step dt (s) for prediction")
    ap.add_argument("--min_dt", type=float, default=0.02, help="Min odom interval to use (s)")
    ap.add_argument("--max_dt", type=float, default=0.2, help="Max odom interval to use (s)")
    ap.add_argument("--motion_gate_w", type=float, default=0.02, help="Min |dyaw| (rad) to keep a segment")
    ap.add_argument("--motion_gate_v", type=float, default=0.02, help="Min sqrt(dx^2+dy^2) (m) to keep a segment")

    # Initial values
    ap.add_argument("--pgl_x0", type=float, default=-0.082)
    ap.add_argument("--pgl_y0", type=float, default=0.168)
    ap.add_argument("--pbg_x0", type=float, default=0.0)
    ap.add_argument("--pbg_y0", type=float, default=0.0)
    ap.add_argument("--dpsi0", type=float, default=0.0)

    # Prior std (set to inf to disable)
    ap.add_argument("--pgl_std", type=float, default=0.03, help="Prior std for p_gl (m)")
    ap.add_argument("--pbg_std", type=float, default=0.05, help="Prior std for p_bg (m)")
    ap.add_argument("--dpsi_std", type=float, default=math.radians(5.0), help="Prior std for dpsi (rad)")

    ap.add_argument("--huber", type=float, default=1.0, help="Huber loss scale for least_squares")
    args = ap.parse_args()

    odoms, chass = read_bag(args.bag, args.odom_topic, args.chassis_topic)
    interp = ChassisInterp(chass)

    x0 = np.array([args.pgl_x0, args.pgl_y0, args.pbg_x0, args.pbg_y0, args.dpsi0], dtype=np.float64)

    prior_mean = np.array([args.pgl_x0, args.pgl_y0, args.pbg_x0, args.pbg_y0, args.dpsi0], dtype=np.float64)
    prior_std = np.array([args.pgl_std, args.pgl_std, args.pbg_std, args.pbg_std, args.dpsi_std], dtype=np.float64)

    def fun(x):
        return build_residuals(
            x, odoms, interp,
            step_dt=args.step_dt,
            min_dt=args.min_dt,
            max_dt=args.max_dt,
            motion_gate_w=args.motion_gate_w,
            motion_gate_v=args.motion_gate_v,
            prior=(prior_mean, prior_std)
        )

    # Optimize
    res = least_squares(fun, x0, loss='huber', f_scale=args.huber, verbose=2, max_nfev=200)

    x_opt = res.x
    print("\n===== Calibration Result (SE2) =====")
    print(f"p_gl_xy = [{x_opt[0]:+.6f}, {x_opt[1]:+.6f}]  (m)")
    print(f"p_bg_xy = [{x_opt[2]:+.6f}, {x_opt[3]:+.6f}]  (m)")
    print(f"R_gl yaw (dpsi) = {x_opt[4]:+.6f} rad  ({math.degrees(x_opt[4]):+.3f} deg)")
    print(f"Cost = {res.cost:.6f}, success={res.success}, msg={res.message}")


if __name__ == "__main__":
    main()


# python3 offline_extrinsic_calib_se2.py \
#   --bag ~/bags/run1 \
#   --odom_topic /Odomtry \
#   --chassis_topic /chassis_info \
#   --pgl_x0 -0.082 --pgl_y0 0.168 \
#   --pbg_x0 0.0 --pbg_y0 0.0 \
#   --dpsi0 0.0

# #!/usr/bin/env python3
# import math
# from collections import deque
# from typing import Deque, Tuple

# import rclpy
# from rclpy.node import Node
# from nav_msgs.msg import Odometry
# from std_msgs.msg import Float64


# class OdomAvgSpeedRadiusNode(Node):
#     def __init__(self):
#         super().__init__('odom_avg_speed_radius')

#         # ---- parameters ----
#         self.declare_parameter('in_topic', '/Odometry')
#         self.declare_parameter('window_sec', 10.0)       # 平均时间窗（秒）
#         self.declare_parameter('min_abs_omega', 0.5)   # 角速度阈值
#         self.declare_parameter('min_samples', 5)        # 窗内最少样本才输出
#         self.declare_parameter('publish_rate_hz', 4.0) # 定时发布平均结果
#         self.declare_parameter('log_every_publish', 10)  # 0=不打印；例如10=每10次发布打印

#         self.in_topic = str(self.get_parameter('in_topic').value)
#         self.window_sec = float(self.get_parameter('window_sec').value)
#         self.min_abs_omega = float(self.get_parameter('min_abs_omega').value)
#         self.min_samples = int(self.get_parameter('min_samples').value)
#         self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
#         self.log_every_publish = int(self.get_parameter('log_every_publish').value)

#         # buffer: (t_sec, vxy, radius_or_nan)
#         self.buf: Deque[Tuple[float, float, float]] = deque()

#         self.sub = self.create_subscription(Odometry, self.in_topic, self.cb_odom, 50)

#         self.pub_speed = self.create_publisher(Float64, 'speed_xy_avg', 10)
#         self.pub_radius = self.create_publisher(Float64, 'radius_avg', 10)

#         period = 1.0 / max(1e-6, self.publish_rate_hz)
#         self.timer = self.create_timer(period, self.on_timer)

#         self.publish_count = 0

#         self.get_logger().info(
#             f"Subscribed: {self.in_topic}\n"
#             f"Publishing: ~speed_xy_avg, ~radius_avg\n"
#             f"window_sec={self.window_sec}, min_abs_omega={self.min_abs_omega}, "
#             f"min_samples={self.min_samples}, publish_rate_hz={self.publish_rate_hz}"
#         )

#     def cb_odom(self, msg: Odometry):
#         vx = float(msg.twist.twist.linear.x)
#         vy = float(msg.twist.twist.linear.y)
#         wz = float(msg.twist.twist.angular.z)

#         vxy = math.sqrt(vx * vx + vy * vy)

#         if abs(wz) >= self.min_abs_omega:
#             radius = vxy / abs(wz)
#         else:
#             radius = float('nan')

#         t = self.get_clock().now().nanoseconds * 1e-9
#         self.buf.append((t, vxy, radius))

#         # prune old samples
#         t_min = t - self.window_sec
#         while self.buf and self.buf[0][0] < t_min:
#             self.buf.popleft()

#     def on_timer(self):
#         if len(self.buf) < self.min_samples:
#             return

#         # average vxy over all samples in window
#         v_sum = 0.0
#         n = 0

#         # average radius only over valid (non-NaN) samples
#         r_sum = 0.0
#         rn = 0

#         for _, vxy, r in self.buf:
#             v_sum += vxy
#             n += 1
#             if not math.isnan(r) and math.isfinite(r):
#                 r_sum += r
#                 rn += 1

#         v_avg = v_sum / max(1, n)
#         r_avg = (r_sum / rn) if rn > 0 else float('nan')

#         m1 = Float64()
#         m1.data = v_avg
#         self.pub_speed.publish(m1)

#         m2 = Float64()
#         m2.data = r_avg
#         self.pub_radius.publish(m2)

#         self.publish_count += 1
#         if self.log_every_publish > 0 and (self.publish_count % self.log_every_publish == 0):
#             # window actual duration
#             t0 = self.buf[0][0]
#             t1 = self.buf[-1][0]
#             self.get_logger().info(
#                 f"[win={t1 - t0:.2f}s, n={n}, rn={rn}] v_avg={v_avg:.4f} m/s, r_avg={r_avg:.4f} m"
#             )


# def main():
#     rclpy.init()
#     node = OdomAvgSpeedRadiusNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
import math
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Float64


def fit_circle_kasa(points_xy: np.ndarray) -> Tuple[float, float, float]:
    """
    Algebraic least-squares circle fit (Kåsa).
    Solve: x^2 + y^2 = D*x + E*y + F
    center = (D/2, E/2), r = sqrt(center^2 + F)

    points_xy: (N,2)
    returns: (cx, cy, r)
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y

    # least squares
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cx = 0.5 * D
    cy = 0.5 * E
    r2 = cx * cx + cy * cy + F
    r = math.sqrt(max(0.0, float(r2)))
    return float(cx), float(cy), float(r)


def circle_residuals(points_xy: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    d = np.sqrt((points_xy[:, 0] - cx) ** 2 + (points_xy[:, 1] - cy) ** 2)
    return d - r


def fit_circle_trimmed(points_xy: np.ndarray, trim_ratio: float, iters: int = 1) -> Tuple[float, float, float, float]:
    """
    Fit circle, then optionally trim worst residuals and refit.
    Returns (cx, cy, r, rmse)
    """
    pts = points_xy
    cx, cy, r = fit_circle_kasa(pts)

    # iterative trimming
    for _ in range(max(0, iters)):
        if trim_ratio <= 0.0 or len(pts) < 10:
            break
        res = circle_residuals(pts, cx, cy, r)
        abs_res = np.abs(res)

        keep_n = int(max(3, round((1.0 - trim_ratio) * len(pts))))
        if keep_n >= len(pts):
            break

        idx = np.argpartition(abs_res, keep_n - 1)[:keep_n]
        pts = pts[idx]
        cx, cy, r = fit_circle_kasa(pts)

    # rmse on original points
    res_all = circle_residuals(points_xy, cx, cy, r)
    rmse = float(np.sqrt(np.mean(res_all ** 2))) if len(points_xy) > 0 else float('nan')
    return cx, cy, r, rmse


class LioPathRadiusFitter(Node):
    def __init__(self):
        super().__init__('lio_path_radius_fitter')

        # ---- parameters ----
        self.declare_parameter('in_topic', '/path')          # 你的 LIO Path 话题名
        self.declare_parameter('max_points', 2000)           # 缓存最多点数（多圈没问题，但别无限涨）
        self.declare_parameter('min_points', 50)             # 少于该点数不拟合
        self.declare_parameter('fit_rate_hz', 0.5)           # 拟合频率
        self.declare_parameter('trim_ratio', 0.1)            # 丢弃最大残差比例（0~0.4建议）
        self.declare_parameter('trim_iters', 10)              # trim 迭代次数
        self.declare_parameter('min_step_dist', 0.01)        # 新增点距离阈值（m），太密会影响数值稳定/效率
        self.declare_parameter('log_every_fit', 1)           # 每次拟合都打印=1；0=不打印；>1=每N次打印

        self.in_topic = str(self.get_parameter('in_topic').value)
        self.max_points = int(self.get_parameter('max_points').value)
        self.min_points = int(self.get_parameter('min_points').value)
        self.fit_rate_hz = float(self.get_parameter('fit_rate_hz').value)
        self.trim_ratio = float(self.get_parameter('trim_ratio').value)
        self.trim_iters = int(self.get_parameter('trim_iters').value)
        self.min_step_dist = float(self.get_parameter('min_step_dist').value)
        self.log_every_fit = int(self.get_parameter('log_every_fit').value)

        self.points: Deque[Tuple[float, float]] = deque(maxlen=max(10, self.max_points))
        self.last_path_len = 0
        self.last_xy: Optional[Tuple[float, float]] = None

        self.sub = self.create_subscription(Path, self.in_topic, self.cb_path, 10)

        # publish results
        self.pub_radius = self.create_publisher(Float64, 'radius_fit', 10)
        self.pub_rmse = self.create_publisher(Float64, 'radius_fit_rmse', 10)

        self.fit_count = 0
        period = 1.0 / max(1e-6, self.fit_rate_hz)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"Subscribed: {self.in_topic}\n"
            f"Publishing: ~radius_fit, ~radius_fit_rmse\n"
            f"max_points={self.max_points}, min_points={self.min_points}, fit_rate_hz={self.fit_rate_hz}, "
            f"trim_ratio={self.trim_ratio}, trim_iters={self.trim_iters}, min_step_dist={self.min_step_dist}"
        )

    def cb_path(self, msg: Path):
        n = len(msg.poses)
        if n == 0:
            return

        # 如果 path 被重置/截断（比如重新建图），就清空
        if n < self.last_path_len:
            self.points.clear()
            self.last_path_len = 0
            self.last_xy = None

        # 增量取新增 poses
        for i in range(self.last_path_len, n):
            p = msg.poses[i].pose.position
            x, y = float(p.x), float(p.y)

            if self.last_xy is not None:
                dx = x - self.last_xy[0]
                dy = y - self.last_xy[1]
                if math.hypot(dx, dy) < self.min_step_dist:
                    continue

            self.points.append((x, y))
            self.last_xy = (x, y)

        self.last_path_len = n

    def on_timer(self):
        if len(self.points) < self.min_points:
            return

        pts = np.array(self.points, dtype=float)  # (N,2)
        cx, cy, r, rmse = fit_circle_trimmed(pts, trim_ratio=self.trim_ratio, iters=self.trim_iters)

        # publish
        m_r = Float64()
        m_r.data = float(r)
        self.pub_radius.publish(m_r)

        m_e = Float64()
        m_e.data = float(rmse)
        self.pub_rmse.publish(m_e)

        self.fit_count += 1
        if self.log_every_fit > 0 and (self.fit_count % self.log_every_fit == 0):
            self.get_logger().info(
                f"[N={len(self.points)}] radius={r:.4f} m, center=({cx:.4f},{cy:.4f}), rmse={rmse:.4f} m"
            )


def main():
    rclpy.init()
    node = LioPathRadiusFitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

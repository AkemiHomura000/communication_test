#!/usr/bin/env python3
"""
icp_pcd_match.py
================
对两个 .pcd 文件做 ICP 点云配准，输出变换结果并可视化。

说明
----
  source  : 待变换点云（例如第二个激光雷达的点云）
  target  : 参考点云（例如第一个激光雷达的点云，或地图）

  ICP 求解的变换 T 满足：
      T * source  ≈  target
  即把 source 配准到 target 坐标系下。

  输出
  ----
  · 4×4 变换矩阵 T
  · 平移向量 [tx, ty, tz]（单位 m）
  · 旋转欧拉角 [roll, pitch, yaw]（ZYX 约定，单位 deg）
  · ICP fitness score（越小越好）

  可视化
  ------
  绿色 = source 配准前（原始位置）
  蓝色 = source 配准后（ICP 变换后）
  红色 = target（参考点云）

依赖
----
  pip install open3d numpy scipy
"""

# ===========================================================================
#  ★★★  用户配置区  ★★★  —— 所有参数均在此处修改，无需改动下方代码
# ===========================================================================

# ---------- 文件路径 ----------
SOURCE_PCD = "/home/rm/Desktop/communication_test/src/tools/lidar_extrinsic_calibration/pcd_output/livox_lidar_192_168_1_133_integrated_100frames.pcd"   # 待配准点云路径（第二个激光雷达）
TARGET_PCD = "/home/rm/Desktop/communication_test/src/tools/lidar_extrinsic_calibration/pcd_output/livox_lidar_192_168_1_183_integrated_100frames.pcd"   # 参考点云路径（第一个激光雷达 / 地图）

# ---------- 初始粗变换（平移 + 欧拉角） ----------
# 根据先验知识手动填写，用于给 ICP 提供良好初值。
# 若两帧点云已近似对齐，保持全零即可。
#
# 旋转约定：ZYX 外旋（即先绕 Z 转 yaw，再绕 Y 转 pitch，再绕 X 转 roll）
# 单位：平移 m，角度 deg
INIT_TX    =  0.0   # 沿 X 轴平移（m）
INIT_TY    =  0.0   # 沿 Y 轴平移（m）
INIT_TZ    =  0.0   # 沿 Z 轴平移（m）
INIT_ROLL  =  90.0   # 绕 X 轴旋转，roll（deg）
INIT_PITCH =  0.0   # 绕 Y 轴旋转，pitch（deg）
INIT_YAW   =  0.0   # 绕 Z 轴旋转，yaw（deg）

# ---------- ICP 参数 ----------
# 配准方法：
#   "point_to_point"  —— 点对点（最通用，不需要法线）
#   "point_to_plane"  —— 点对面（精度更高，需要估计法线）
#   "generalized"     —— 广义 ICP（鲁棒性强，需要估计法线）
ICP_METHOD   = "point_to_plane"

VOXEL_SIZE   = 0.05    # 体素降采样大小（m），0 表示不降采样
ICP_DIST     = 0.5     # ICP 最大对应点距离阈值（m）
ICP_MAX_ITER = 200     # ICP 最大迭代次数

# ---------- 可视化 ----------
NO_VIS = False   # True = 不弹出可视化窗口

# ===========================================================================
#  以下为功能实现代码，通常无需修改
# ===========================================================================

import math
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] 未找到 open3d，请先安装：pip install open3d")
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as ScipyRot
    _USE_SCIPY = True
except ImportError:
    _USE_SCIPY = False


# ---------------------------------------------------------------------------
# 欧拉角提取 (ZYX: yaw-pitch-roll)
# ---------------------------------------------------------------------------

def rotmat_to_euler_zyx(R: np.ndarray):
    """
    从旋转矩阵提取 ZYX 欧拉角 (yaw, pitch, roll)。
    返回 (roll, pitch, yaw)，单位弧度。
    使用 scipy 优先；无 scipy 时回退到手写公式（gimbal-lock 附近精度下降）。
    """
    if _USE_SCIPY:
        r = ScipyRot.from_matrix(R[:3, :3])
        # 'ZYX' 顺序对应 extrinsic x-y-z = intrinsic Z-Y-X
        rpy = r.as_euler('ZYX', degrees=False)   # [yaw, pitch, roll]
        return float(rpy[2]), float(rpy[1]), float(rpy[0])  # roll, pitch, yaw

    # 手写 ZYX 分解
    R33 = R[:3, :3]
    sy = math.sqrt(R33[0, 0] ** 2 + R33[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2( R33[2, 1], R33[2, 2])
        pitch = math.atan2(-R33[2, 0], sy)
        yaw   = math.atan2( R33[1, 0], R33[0, 0])
    else:
        roll  = math.atan2(-R33[1, 2], R33[1, 1])
        pitch = math.atan2(-R33[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# 打印结果
# ---------------------------------------------------------------------------

def print_result(T: np.ndarray, fitness: float, inlier_rmse: float, n_pairs: int):
    roll, pitch, yaw = rotmat_to_euler_zyx(T)
    tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  ICP 配准结果")
    print(sep)
    print(f"  fitness score   : {fitness:.6f}  (越大越好，最大 1.0)")
    print(f"  inlier RMSE     : {inlier_rmse:.6f} m")
    print(f"  对应点对数       : {n_pairs}")
    print(sep)
    print(f"  平移向量 (m)")
    print(f"    tx = {tx:+.6f}")
    print(f"    ty = {ty:+.6f}")
    print(f"    tz = {tz:+.6f}")
    print(sep)
    print(f"  旋转欧拉角 ZYX (deg)")
    print(f"    roll  (X) = {math.degrees(roll):+.4f}°")
    print(f"    pitch (Y) = {math.degrees(pitch):+.4f}°")
    print(f"    yaw   (Z) = {math.degrees(yaw):+.4f}°")
    print(sep)
    print(f"  旋转欧拉角 ZYX (rad)")
    print(f"    roll  (X) = {roll:+.6f}")
    print(f"    pitch (Y) = {pitch:+.6f}")
    print(f"    yaw   (Z) = {yaw:+.6f}")
    print(sep)
    print(f"  4×4 变换矩阵 T  (T * source ≈ target)")
    for row in T:
        print(f"    [{row[0]:+10.6f}  {row[1]:+10.6f}  {row[2]:+10.6f}  {row[3]:+10.6f}]")
    print(sep)

    # 便于复制到配置文件的单行格式
    print(f"\n  一行格式（可直接粘贴）：")
    print(f"    tx={tx:+.6f}  ty={ty:+.6f}  tz={tz:+.6f}  "
          f"roll={math.degrees(roll):+.4f}deg  "
          f"pitch={math.degrees(pitch):+.4f}deg  "
          f"yaw={math.degrees(yaw):+.4f}deg")
    print()


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize(src_orig: o3d.geometry.PointCloud,
              src_aligned: o3d.geometry.PointCloud,
              tgt: o3d.geometry.PointCloud,
              title: str = "ICP 配准结果"):
    """
    绿色 = source 原始位置
    蓝色 = source 配准后（ICP 变换后）
    红色 = target（参考点云）
    """
    src_orig_vis = o3d.geometry.PointCloud(src_orig)
    src_alig_vis = o3d.geometry.PointCloud(src_aligned)
    tgt_vis      = o3d.geometry.PointCloud(tgt)

    # 对红/绿点云额外降采样，使其视觉上更稀疏，从而突出蓝色配准结果
    _bg_voxel = 0.08   # 背景点云降采样体素（调大则更稀疏）
    src_orig_vis = src_orig_vis.voxel_down_sample(_bg_voxel)
    tgt_vis      = tgt_vis.voxel_down_sample(_bg_voxel)

    src_orig_vis.paint_uniform_color([0.0, 0.75, 0.0])  # 绿（稀疏背景）
    src_alig_vis.paint_uniform_color([0.0, 0.5, 1.0])   # 蓝（配准后，完整密度）
    tgt_vis.paint_uniform_color([1.0, 0.2, 0.2])         # 红（稀疏背景）

    # 坐标轴（原点处）
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    print("  [可视化窗口]")
    print("    绿色(稀疏) = source 原始位置")
    print("    蓝色(完整) = source ICP 配准后  ← 主要观察对象")
    print("    红色(稀疏) = target（参考点云）")
    print(f"    红/绿降采体素={_bg_voxel} m，如需调整请修改脚本中的 _bg_voxel")
    print("    关闭窗口后程序退出。\n")

    o3d.visualization.draw_geometries(
        [src_orig_vis, src_alig_vis, tgt_vis, frame],
        window_name=title,
        width=1280,
        height=800,
        point_show_normal=False,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_icp(src_path: str,
            tgt_path: str,
            voxel_size: float,
            icp_dist: float,
            icp_max_iter: int,
            method: str,
            init_T: np.ndarray,
            no_vis: bool):

    # ---- 读取 ----
    print(f"[1/4] 读取点云")
    print(f"      source : {src_path}")
    print(f"      target : {tgt_path}")

    src = o3d.io.read_point_cloud(src_path)
    tgt = o3d.io.read_point_cloud(tgt_path)

    if len(src.points) == 0:
        print(f"[ERROR] source 点云为空，请检查路径：{src_path}")
        sys.exit(1)
    if len(tgt.points) == 0:
        print(f"[ERROR] target 点云为空，请检查路径：{tgt_path}")
        sys.exit(1)

    print(f"      source 点数: {len(src.points)}")
    print(f"      target 点数: {len(tgt.points)}")

    # ---- 降采样（point-to-plane / generalized 使用降采后点云） ----
    if voxel_size > 0:
        print(f"\n[2/4] 体素降采样 (voxel_size={voxel_size} m)")
        src_down = src.voxel_down_sample(voxel_size)
        tgt_down = tgt.voxel_down_sample(voxel_size)
        print(f"      source 降采后: {len(src_down.points)} 点")
        print(f"      target 降采后: {len(tgt_down.points)} 点")
    else:
        src_down = src
        tgt_down = tgt

    # ---- 使用手动设置的初始粗变换 ----
    T_init = init_T.copy()
    print(f"\n[2/4] 使用手动设置的初始粗变换矩阵（见顶部 INIT_T）")

    # ---- ICP 精配准 ----
    print(f"\n[3/4] ICP 精配准 (method={method}, max_dist={icp_dist} m, max_iter={icp_max_iter})")

    if method == "point_to_plane":
        # point-to-plane 需要法线
        radius_n = max(icp_dist * 2, 0.1)
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        icp_src, icp_tgt = src_down, tgt_down
    elif method == "generalized":
        radius_n = max(icp_dist * 2, 0.1)
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=30))
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        icp_src, icp_tgt = src_down, tgt_down
    else:  # point_to_point
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        icp_src, icp_tgt = src, tgt   # point-to-point 用原始密度更精准

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-7,
        relative_rmse=1e-7,
        max_iteration=icp_max_iter,
    )

    result = o3d.pipelines.registration.registration_icp(
        icp_src, icp_tgt,
        icp_dist,
        T_init,
        estimation,
        criteria,
    )

    T = result.transformation
    fitness = result.fitness
    inlier_rmse = result.inlier_rmse
    n_pairs = len(result.correspondence_set)

    # ---- 打印结果 ----
    print(f"\n[4/4] 结果输出")
    print_result(T, fitness, inlier_rmse, n_pairs)

    # ---- 可视化 ----
    if not no_vis:
        src_aligned = o3d.geometry.PointCloud(src)
        src_aligned.transform(T)
        visualize(src, src_aligned, tgt,
                  title=f"ICP: {src_path.split('/')[-1]}  →  {tgt_path.split('/')[-1]}")

    return T, result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 将欧拉角（deg）+ 平移向量 转换为 4×4 变换矩阵
    import math as _math
    _roll  = _math.radians(INIT_ROLL)
    _pitch = _math.radians(INIT_PITCH)
    _yaw   = _math.radians(INIT_YAW)

    # ZYX 旋转矩阵：R = Rz(yaw) * Ry(pitch) * Rx(roll)
    _cr, _sr = _math.cos(_roll),  _math.sin(_roll)
    _cp, _sp = _math.cos(_pitch), _math.sin(_pitch)
    _cy, _sy = _math.cos(_yaw),   _math.sin(_yaw)

    init_T = np.array([
        [_cy*_cp,  _cy*_sp*_sr - _sy*_cr,  _cy*_sp*_cr + _sy*_sr,  INIT_TX],
        [_sy*_cp,  _sy*_sp*_sr + _cy*_cr,  _sy*_sp*_cr - _cy*_sr,  INIT_TY],
        [   -_sp,             _cp*_sr,                 _cp*_cr,     INIT_TZ],
        [    0.0,                 0.0,                     0.0,        1.0  ],
    ], dtype=np.float64)

    run_icp(
        src_path=SOURCE_PCD,
        tgt_path=TARGET_PCD,
        voxel_size=VOXEL_SIZE,
        icp_dist=ICP_DIST,
        icp_max_iter=ICP_MAX_ITER,
        method=ICP_METHOD,
        init_T=init_T,
        no_vis=NO_VIS,
    )

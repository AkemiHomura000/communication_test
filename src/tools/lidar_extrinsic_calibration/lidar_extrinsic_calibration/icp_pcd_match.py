#!/usr/bin/env python3
"""
icp_pcd_match.py
================
对两个 .pcd 文件做 ICP 点云配准，输出变换结果并可视化。

用法
----
python3 icp_pcd_match.py <source.pcd> <target.pcd> [选项]

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

import argparse
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
# 点云预处理
# ---------------------------------------------------------------------------

def preprocess(pcd: o3d.geometry.PointCloud,
               voxel_size: float) -> tuple:
    """
    体素降采样 + 法线估计 + FPFH 特征提取。
    返回 (pcd_down, fpfh)
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh


def global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh,
                         voxel_size: float) -> o3d.pipelines.registration.RegistrationResult:
    """
    基于 FPFH 的快速全局配准（RANSAC），用于给 ICP 提供初始变换。
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


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
            use_global: bool,
            init_T: np.ndarray,
            no_vis: bool):

    # ---- 读取 ----
    print(f"[1/5] 读取点云")
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

    # ---- 降采样（仅用于全局配准和 point-to-plane ICP） ----
    if voxel_size > 0:
        print(f"\n[2/5] 体素降采样 (voxel_size={voxel_size} m)")
        src_down = src.voxel_down_sample(voxel_size)
        tgt_down = tgt.voxel_down_sample(voxel_size)
        print(f"      source 降采后: {len(src_down.points)} 点")
        print(f"      target 降采后: {len(tgt_down.points)} 点")
    else:
        src_down = src
        tgt_down = tgt

    # ---- 全局粗配准（可选） ----
    T_init = init_T.copy()
    if use_global:
        print(f"\n[3/5] 全局粗配准 (FPFH + RANSAC) …")
        if voxel_size <= 0:
            gs = 0.05
            src_gs = src.voxel_down_sample(gs)
            tgt_gs = tgt.voxel_down_sample(gs)
        else:
            gs = voxel_size
            src_gs, tgt_gs = src_down, tgt_down

        src_gs_d, src_fpfh = preprocess(src_gs, gs)
        tgt_gs_d, tgt_fpfh = preprocess(tgt_gs, gs)

        global_res = global_registration(src_gs_d, tgt_gs_d, src_fpfh, tgt_fpfh, gs)
        T_init = global_res.transformation
        print(f"      全局配准 fitness={global_res.fitness:.4f}")
    else:
        print(f"\n[3/5] 跳过全局粗配准（使用单位矩阵或指定初始变换）")

    # ---- ICP 精配准 ----
    print(f"\n[4/5] ICP 精配准 (method={method}, max_dist={icp_dist} m, max_iter={icp_max_iter})")

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
    print(f"\n[5/5] 结果输出")
    print_result(T, fitness, inlier_rmse, n_pairs)

    # ---- 可视化 ----
    if not no_vis:
        src_aligned = o3d.geometry.PointCloud(src)
        src_aligned.transform(T)
        visualize(src, src_aligned, tgt,
                  title=f"ICP: {src_path.split('/')[-1]}  →  {tgt_path.split('/')[-1]}")

    return T, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ICP 点云配准工具：将 source.pcd 配准到 target.pcd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例
----
# 基本用法（point-to-point ICP，开启全局粗配准）
python3 icp_pcd_match.py src.pcd tgt.pcd

# 使用 point-to-plane ICP，指定体素大小和最大对应距离
python3 icp_pcd_match.py src.pcd tgt.pcd --method point_to_plane --voxel 0.05 --dist 0.1

# 不做全局配准（已知初始位置接近），不弹出可视化窗口
python3 icp_pcd_match.py src.pcd tgt.pcd --no_global --no_vis
        """,
    )

    parser.add_argument("source",        help="source 点云文件路径（待配准）")
    parser.add_argument("target",        help="target 点云文件路径（参考/地图）")

    parser.add_argument(
        "--method", default="point_to_point",
        choices=["point_to_point", "point_to_plane", "generalized"],
        help="ICP 配准方法（默认 point_to_point）",
    )
    parser.add_argument(
        "--voxel", type=float, default=0.05,
        help="体素降采样大小（m），0 表示不降采样（默认 0.05）",
    )
    parser.add_argument(
        "--dist", type=float, default=0.5,
        help="ICP 最大对应点距离阈值（m，默认 0.5）",
    )
    parser.add_argument(
        "--max_iter", type=int, default=200,
        help="ICP 最大迭代次数（默认 200）",
    )
    parser.add_argument(
        "--no_global", action="store_true",
        help="跳过全局粗配准（RANSAC），直接用单位矩阵初始化",
    )
    parser.add_argument(
        "--no_vis", action="store_true",
        help="不弹出可视化窗口",
    )
    parser.add_argument(
        "--init_T", type=float, nargs=16, default=None,
        metavar="T_ij",
        help="自定义 4×4 初始变换矩阵（行优先展开的 16 个数），会覆盖全局配准结果",
    )

    args = parser.parse_args()

    # 构造初始变换矩阵
    if args.init_T is not None:
        init_T = np.array(args.init_T, dtype=np.float64).reshape(4, 4)
        use_global = False   # 指定了初始值则不做全局配准
    else:
        init_T = np.eye(4, dtype=np.float64)
        use_global = not args.no_global

    run_icp(
        src_path=args.source,
        tgt_path=args.target,
        voxel_size=args.voxel,
        icp_dist=args.dist,
        icp_max_iter=args.max_iter,
        method=args.method,
        use_global=use_global,
        init_T=init_T,
        no_vis=args.no_vis,
    )


if __name__ == "__main__":
    main()

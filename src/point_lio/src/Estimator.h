#ifndef Estimator_H
#define Estimator_H

#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <unordered_set>

// =====================================================================
// Estimator — 封装 EKF 状态、点云工作缓冲区及观测模型
// =====================================================================
class Estimator
{
public:
  Estimator();
  ~Estimator() = default;

  // ── EKF 实例 ──────────────────────────────────────────────────────
  esekfom::esekf<state_output, 30, input_ikfom> kf_output;
  input_ikfom input_in;

  // ── 姿态辅助 ──────────────────────────────────────────────────────
  V3D    Lidar_T_wrt_IMU {Zero3d};
  M3D    Lidar_R_wrt_IMU {Eye3d};
  double G_m_s2 = 9.81;

  // ── 点云工作缓冲区 ────────────────────────────────────────────────
  PointCloudXYZI::Ptr normvec         {new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr feats_down_body {new PointCloudXYZI(10000,  1)};
  PointCloudXYZI::Ptr feats_down_world{new PointCloudXYZI(10000,  1)};
  std::vector<V3D>         pbody_list;
  std::vector<PointVector> Nearest_Points;
  std::vector<float>       pointSearchSqDis;
  bool                     point_selected_surf[100000] = {};
  std::vector<M3D>         crossmat_list;
  int                      effct_feat_num  = 0;
  int                      feats_down_size = 0;

  // ── 逐点迭代索引 ──────────────────────────────────────────────────
  int k   = 0;
  int idx = -1;

  // ── IMU 测量均值（供观测模型使用）────────────────────────────────
  V3D angvel_avr   {V3D::Zero()};
  V3D acc_avr      {V3D::Zero()};
  V3D acc_avr_norm {V3D::Zero()};

  // ── IVox 地图 ─────────────────────────────────────────────────────
  std::shared_ptr<IVoxType> ivox_ = nullptr;

  // ── 时序 ──────────────────────────────────────────────────────────
  std::vector<int> time_seq;

  // ── 初始化 ────────────────────────────────────────────────────────
  void initKalmanFilter(const Eigen::Matrix<double, 30, 30> &P_init,
                        const Eigen::Matrix<double, 30, 30> &Q);

  // ── 坐标变换 ──────────────────────────────────────────────────────
  void pointBodyToWorld(PointType const *pi, PointType *po) const;

  // ── 静态噪声/状态模型（供 esekf 函数指针使用）────────────────────
  static Eigen::Matrix<double, 30, 30> process_noise_cov_output();
  static Eigen::Matrix<double, 30, 1>  get_f_output(state_output &s, const input_ikfom &in);
  static Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

  // ── 观测模型（成员函数，需访问实例缓冲区）────────────────────────
  void h_model_output    (state_output &s,
                          Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R,
                          esekfom::dyn_share_modified<double> &ekfom_data);
  void h_model_IMU_output(state_output &s,
                          esekfom::dyn_share_modified<double> &ekfom_data);
};

// ── 全局单例指针（仅供 esekf 回调桥接使用）──────────────────────────
extern Estimator *g_estimator;

// ── 自由函数桥接（转发到 g_estimator->*）────────────────────────────
void h_model_output_bridge    (state_output &s,
                                Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R,
                                esekfom::dyn_share_modified<double> &ekfom_data);
void h_model_IMU_output_bridge(state_output &s,
                                esekfom::dyn_share_modified<double> &ekfom_data);

#endif // Estimator_H

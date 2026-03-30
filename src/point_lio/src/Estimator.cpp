#include "Estimator.h"

// ── 全局单例指针（由 LaserMappingNode 构造时设置）──────────────────
Estimator *g_estimator = nullptr;

// =====================================================================
//  构造函数
// =====================================================================
Estimator::Estimator()
{
  pointSearchSqDis.resize(NUM_MATCH_POINTS);
  std::memset(point_selected_surf, false, sizeof(point_selected_surf));
}

// =====================================================================
//  初始化 KF
// =====================================================================
void Estimator::initKalmanFilter(const Eigen::Matrix<double, 30, 30> &P_init,
                                 const Eigen::Matrix<double, 30, 30> &Q)
{
  kf_output.init_dyn_share_modified_3h(
      get_f_output, df_dx_output,
      h_model_output_bridge, h_model_IMU_output_bridge);
  kf_output.change_P(const_cast<Eigen::Matrix<double, 30, 30> &>(P_init));
}

// =====================================================================
//  坐标变换：LiDAR body → world
// =====================================================================
void Estimator::pointBodyToWorld(PointType const *const pi, PointType *const po) const
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global;
  if (extrinsic_est_en)
  {
    p_global = kf_output.x_.rot *
               (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) +
               kf_output.x_.pos;
  }
  else
  {
    p_global = kf_output.x_.rot *
               (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) +
               kf_output.x_.pos;
  }
  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

// =====================================================================
//  静态：过程噪声协方差
// =====================================================================
Eigen::Matrix<double, 30, 30> Estimator::process_noise_cov_output()
{
  Eigen::Matrix<double, 30, 30> cov;
  cov.setZero();
  cov.block<3, 3>(12, 12).diagonal() << vel_cov,        vel_cov,        vel_cov;
  cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
  cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
  cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov,      b_gyr_cov,      b_gyr_cov;
  cov.block<3, 3>(27, 27).diagonal() << b_acc_cov,      b_acc_cov,      b_acc_cov;
  return cov;
}

// =====================================================================
//  静态：状态微分
// =====================================================================
Eigen::Matrix<double, 30, 1> Estimator::get_f_output(state_output &s, const input_ikfom &in)
{
  Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
  vect3 a_inertial = s.rot * s.acc;
  for (int i = 0; i < 3; i++)
  {
    res(i)      = s.vel[i];
    res(i + 3)  = s.omg[i];
    res(i + 12) = a_inertial[i] + s.gravity[i];
  }
  return res;
}

// =====================================================================
//  静态：状态 Jacobian
// =====================================================================
Eigen::Matrix<double, 30, 30> Estimator::df_dx_output(state_output &s, const input_ikfom &in)
{
  Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
  cov.template block<3, 3>(0,  12) = Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(12,  3) = -s.rot * MTK::hat(s.acc);
  cov.template block<3, 3>(12, 18) =  s.rot;
  cov.template block<3, 3>(12, 21) =  Eigen::Matrix3d::Identity();
  cov.template block<3, 3>(3,  15) =  Eigen::Matrix3d::Identity();
  return cov;
}

// =====================================================================
//  观测模型：LiDAR 点面匹配
// =====================================================================
void Estimator::h_model_output(state_output &s,
                               Eigen::Matrix3d /*cov_p*/, Eigen::Matrix3d /*cov_R*/,
                               esekfom::dyn_share_modified<double> &ekfom_data)
{
  bool match_in_map = false;
  VF(4) pabcd;
  pabcd.setZero();
  normvec->resize(time_seq[k]);
  int effect_num_k = 0;

  for (int j = 0; j < time_seq[k]; j++)
  {
    PointType &point_body_j  = feats_down_body->points[idx + j + 1];
    PointType &point_world_j = feats_down_world->points[idx + j + 1];
    pointBodyToWorld(&point_body_j, &point_world_j);
    V3D   p_body  = pbody_list[idx + j + 1];
    double p_norm = p_body.norm();
    V3D   p_world(point_world_j.x, point_world_j.y, point_world_j.z);

    {
      auto &points_near = Nearest_Points[idx + j + 1];
      ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);

      if (points_near.size() < NUM_MATCH_POINTS)
      {
        point_selected_surf[idx + j + 1] = false;
      }
      else
      {
        point_selected_surf[idx + j + 1] = false;
        if (esti_plane(pabcd, points_near, plane_thr))
        {
          float pd2 = fabs(pabcd(0) * point_world_j.x +
                          pabcd(1) * point_world_j.y +
                          pabcd(2) * point_world_j.z + pabcd(3));
          if (p_norm > match_s * pd2 * pd2)
          {
            point_selected_surf[idx + j + 1] = true;
            normvec->points[j].x              = pabcd(0);
            normvec->points[j].y              = pabcd(1);
            normvec->points[j].z              = pabcd(2);
            normvec->points[j].intensity      = pabcd(3);
            effect_num_k++;
          }
        }
      }
    }
  }

  if (effect_num_k == 0)
  {
    ekfom_data.valid = false;
    return;
  }

  ekfom_data.M_Noise = laser_point_cov;
  ekfom_data.h_x.resize(effect_num_k, 12);
  ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
  ekfom_data.z.resize(effect_num_k);
  int m = 0;

  for (int j = 0; j < time_seq[k]; j++)
  {
    if (point_selected_surf[idx + j + 1])
    {
      V3D norm_vec(normvec->points[j].x,
                   normvec->points[j].y,
                   normvec->points[j].z);
      if (extrinsic_est_en)
      {
        V3D p_body = pbody_list[idx + j + 1];
        M3D p_crossmat, p_imu_crossmat;
        p_crossmat << SKEW_SYM_MATRX(p_body);
        V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
        p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
        V3D C(s.rot.transpose() * norm_vec);
        V3D A(p_imu_crossmat * C);
        V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
        ekfom_data.h_x.block<1, 12>(m, 0)
            << norm_vec(0), norm_vec(1), norm_vec(2),
               VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
      }
      else
      {
        M3D point_crossmat = crossmat_list[idx + j + 1];
        V3D C(s.rot.transpose() * norm_vec);
        V3D A(point_crossmat * C);
        ekfom_data.h_x.block<1, 12>(m, 0)
            << norm_vec(0), norm_vec(1), norm_vec(2),
               VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
      }
      ekfom_data.z(m) =
          -norm_vec(0) * feats_down_world->points[idx + j + 1].x
          -norm_vec(1) * feats_down_world->points[idx + j + 1].y
          -norm_vec(2) * feats_down_world->points[idx + j + 1].z
          - normvec->points[j].intensity;
      m++;
    }
  }
  effct_feat_num += effect_num_k;
}

// =====================================================================
//  观测模型：IMU 偏置更新
// =====================================================================
void Estimator::h_model_IMU_output(state_output &s,
                                   esekfom::dyn_share_modified<double> &ekfom_data)
{
  std::memset(ekfom_data.satu_check, false, 6);
  ekfom_data.z_IMU.block<3, 1>(0, 0) = angvel_avr - s.omg - s.bg;
  ekfom_data.z_IMU.block<3, 1>(3, 0) =
      acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
  ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov,
                      imu_meas_acc_cov,  imu_meas_acc_cov,  imu_meas_acc_cov;

  if (check_satu)
  {
    auto check = [&](int idx_satu, double val, double limit)
    {
      if (fabs(val) >= 0.99 * limit)
      {
        ekfom_data.satu_check[idx_satu] = true;
        ekfom_data.z_IMU(idx_satu)      = 0.0;
      }
    };
    check(0, angvel_avr(0), satu_gyro);
    check(1, angvel_avr(1), satu_gyro);
    check(2, angvel_avr(2), satu_gyro);
    check(3, acc_avr(0),    satu_acc);
    check(4, acc_avr(1),    satu_acc);
    check(5, acc_avr(2),    satu_acc);
  }
}

// =====================================================================
//  桥接函数（esekf 函数指针 → g_estimator 成员函数）
// =====================================================================
void h_model_output_bridge(state_output &s,
                           Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R,
                           esekfom::dyn_share_modified<double> &ekfom_data)
{
  g_estimator->h_model_output(s, cov_p, cov_R, ekfom_data);
}

void h_model_IMU_output_bridge(state_output &s,
                               esekfom::dyn_share_modified<double> &ekfom_data)
{
  g_estimator->h_model_IMU_output(s, ekfom_data);
}

/*
 * SPDX-License-Identifier: GPL-2.0-only
 * G2M-LIO covariance-propagation implementation.
 * Licensing notice added on 2026-10-28.
 * See LICENSE and THIRD_PARTY_NOTICES.md at the repository root.
 */

#ifndef AKF_LIO_COVARIANCE_PROPAGATION_H
#define AKF_LIO_COVARIANCE_PROPAGATION_H

#include <common_lib.h>
#include <use-ikfom.hpp>

namespace g2m_lio {

class CovariancePropagation {
public:
    static Eigen::Matrix3d ComputeLidarMeasurementCovariance(const Eigen::Vector3d&, double, double);
    // 计算从状态到地图点的雅可比矩阵（右不变误差）
    // p_lidar: 点在Lidar坐标系
    // R_wl: 世界到Lidar的旋转矩阵 (rot * offset_R_L_I)
    // p_transformed: 变换后的点 (rot * (offset_R_L_I * p_lidar + offset_T_L_I))
    // R_w: IMU到世界的旋转矩阵 (rot)，用于将体坐标系旋转误差变换到世界系
    static Eigen::Matrix<double, 3, 6> ComputeJacobian(
        const Eigen::Vector3d& p_lidar,
        const Eigen::Matrix3d& R_wl,
        const Eigen::Vector3d& p_transformed,
        const Eigen::Matrix3d& R_w);

    // 传播状态协方差到地图点
    // J: 雅可比矩阵 (3x6)
    // state_cov: 状态协方差 (6x6)
    static Eigen::Matrix3d PropagateStateCovariance(
        const Eigen::Matrix<double, 3, 6>& J,
        const Eigen::Matrix<double, 6, 6>& state_cov);

    // 组合观测协方差和状态协方差
    // obs_cov: LIDAR观测协方差
    // state_prop_cov: 状态传播协方差
    // is_static: 是否静止
    static Eigen::Matrix3d CombineCovariances(
        const Eigen::Matrix3d& obs_cov,
        const Eigen::Matrix3d& state_prop_cov,
        bool is_static);
};

} // namespace g2m_lio

#endif

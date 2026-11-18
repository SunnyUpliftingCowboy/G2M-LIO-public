/*
 * SPDX-License-Identifier: GPL-2.0-only
 * G2M-LIO covariance-propagation implementation.
 * Licensing notice added on 2026-10-28.
 * See LICENSE and THIRD_PARTY_NOTICES.md at the repository root.
 */

#include "covariance_propagation.hpp"
#include "so3_math.h"

namespace g2m_lio {

Eigen::Matrix3d CovariancePropagation::ComputeLidarMeasurementCovariance(const Eigen::Vector3d& p, double range_sigma, double angle_sigma_deg) {
    const double r = std::max(p.norm(), 1e-6); Eigen::Vector3d u = p / r;
    Eigen::Vector3d e1 = u.cross(Eigen::Vector3d::UnitZ());
    if (e1.squaredNorm() < 1e-8) e1 = u.cross(Eigen::Vector3d::UnitX());
    e1.normalize(); Eigen::Vector3d e2 = u.cross(e1).normalized();
    const double a = r * std::sin(angle_sigma_deg * M_PI / 180.0);
    return range_sigma * range_sigma * (u*u.transpose()) + a*a * (e1*e1.transpose()+e2*e2.transpose());
}


Eigen::Matrix<double, 3, 6> CovariancePropagation::ComputeJacobian(
    const Eigen::Vector3d& p_lidar,
    const Eigen::Matrix3d& R_wl,
    const Eigen::Vector3d& p_transformed,
    const Eigen::Matrix3d& R_w) {

    Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();

    // ∂p_world/∂pos = I₃
    J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    // ∂p_world/∂δθ = -[p_transformed]× * R_w  (右不变误差)
    // p_transformed = R_w * (R_L_I * p_lidar + t_L_I)
    J.block<3, 3>(0, 3) = -SKEW_SYM_MATRIX(p_transformed) * R_w;

    return J;
}

Eigen::Matrix3d CovariancePropagation::PropagateStateCovariance(
    const Eigen::Matrix<double, 3, 6>& J,
    const Eigen::Matrix<double, 6, 6>& state_cov) {

    return J * state_cov * J.transpose();
}

Eigen::Matrix3d CovariancePropagation::CombineCovariances(
    const Eigen::Matrix3d& obs_cov,
    const Eigen::Matrix3d& state_prop_cov,
    bool is_static) {

    if (is_static) {
        return obs_cov;  // 静止：只保留观测协方差
    } else {
        // 运动：观测 + 状态传播
        Eigen::Matrix3d combined = obs_cov + state_prop_cov;

        // 确保正定性
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(combined);
        if (es.eigenvalues().minCoeff() < 1e-8) {
            // 添加正则化
            combined += 1e-8 * Eigen::Matrix3d::Identity();
        }

        return combined;
    }
}

} // namespace g2m_lio

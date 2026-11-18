/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#ifndef AKF_LIO_USE_IKFOM_H
#define AKF_LIO_USE_IKFOM_H

#include "IKFoM_toolkit/esekfom/esekfom.hpp"

namespace g2m_lio
{

    typedef MTK::vect<3, double> vect3;
    typedef MTK::SO3<double> SO3;
    typedef MTK::S2<double, 98090, 10000, 1> S2;
    typedef MTK::vect<1, double> vect1;
    typedef MTK::vect<2, double> vect2;

    MTK_BUILD_MANIFOLD(state_ikfom, ((vect3, pos))((SO3, rot))((SO3, offset_R_L_I))((vect3, offset_T_L_I))((vect3, vel))(
                                        (vect3, bg))((vect3, ba))((S2, grav)));

    MTK_BUILD_MANIFOLD(input_ikfom_base, ((vect3, acc))((vect3, gyro)));

    struct input_ikfom : input_ikfom_base
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        using input_ikfom_base::input_ikfom_base;

        // Mean and endpoint differences of rate samples, in rad/s and m/s^2.
        // Keep compensation inside the process model so its bias/noise
        // derivatives use exactly the same measurements as the nominal state.
        vect3 gyro_delta = Eigen::Vector3d::Zero();
        vect3 acc_delta = Eigen::Vector3d::Zero();
        double integration_dt = 0.0;
        bool coning_compensation = false;
        bool sculling_compensation = false;
    };

    struct CorrectedImuInput
    {
        vect3 gyro;
        vect3 acc;
        Eigen::Matrix3d gyro_bias_jacobian = -Eigen::Matrix3d::Identity();
        Eigen::Matrix3d acc_gyro_bias_jacobian = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d acc_bias_jacobian = -Eigen::Matrix3d::Identity();
    };

    inline CorrectedImuInput correct_imu_input(const state_ikfom &s, const input_ikfom &in)
    {
        CorrectedImuInput out;
        const vect3 w = in.gyro - s.bg;
        const vect3 a = in.acc - s.ba;
        const double h = in.integration_dt;
        out.gyro = w;
        out.acc = a;
        if (in.coning_compensation)
        {
            // Linearly interpolated endpoint RATES, not two half-interval
            // increments: dtheta = h*w + h^2/12 * (w_start x w_end).
            out.gyro += (h / 12.0) * w.cross(in.gyro_delta);
            out.gyro_bias_jacobian += (h / 12.0) * MTK::hat(in.gyro_delta);
        }
        if (in.sculling_compensation)
        {
            // Express the integrated specific force in the START body frame
            // used by get_f. The rotation term is needed even for constant
            // rates; predict() otherwise rotates only by the initial attitude.
            out.acc += (h / 2.0) * w.cross(a) +
                       (h / 12.0) * (w.cross(in.acc_delta) - in.gyro_delta.cross(a));
            out.acc_gyro_bias_jacobian = (h / 2.0) * MTK::hat(a) +
                                         (h / 12.0) * MTK::hat(in.acc_delta);
            out.acc_bias_jacobian += -(h / 2.0) * MTK::hat(w) +
                                     (h / 12.0) * MTK::hat(in.gyro_delta);
        }
        return out;
    }

    MTK_BUILD_MANIFOLD(process_noise_ikfom, ((vect3, ng))((vect3, na))((vect3, nbg))((vect3, nba)));

    inline MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
    {
        MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
        MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001); // 0.03
        MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na,
                                                        0.0001); // *dt 0.01 0.01 * dt * dt 0.05
        MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg,
                                                        0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
        MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba,
                                                        0.00001); // 0.001 0.05 0.0001/out 0.01
        return cov;
    }

    inline Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
        const auto corrected = correct_imu_input(s, in);
        vect3 a_inertial = s.rot * corrected.acc;
        for (int i = 0; i < 3; i++)
        {
            res(i) = s.vel[i];
            res(i + 3) = corrected.gyro[i];
            res(i + 12) = a_inertial[i] + s.grav[i];
        }
        return res;
    }

    inline Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
        cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
        const auto corrected = correct_imu_input(s, in);
        const Eigen::Matrix3d rotation = s.rot.toRotationMatrix();
        cov.template block<3, 3>(12, 3) = -rotation * MTK::hat(corrected.acc);
        cov.template block<3, 3>(12, 15) = rotation * corrected.acc_gyro_bias_jacobian;
        cov.template block<3, 3>(12, 18) = rotation * corrected.acc_bias_jacobian;
        Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
        Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
        s.S2_Mx(grav_matrix, vec, 21);
        cov.template block<3, 2>(12, 21) = grav_matrix;
        cov.template block<3, 3>(3, 15) = corrected.gyro_bias_jacobian;
        return cov;
    }

    inline Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
    {
        Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
        const auto corrected = correct_imu_input(s, in);
        // Retain the existing interval-rate noise convention: a common
        // additive measurement error at both endpoints, like a bias offset.
        cov.template block<3, 3>(12, 0) = s.rot.toRotationMatrix() * corrected.acc_gyro_bias_jacobian;
        cov.template block<3, 3>(12, 3) = s.rot.toRotationMatrix() * corrected.acc_bias_jacobian;
        cov.template block<3, 3>(3, 0) = corrected.gyro_bias_jacobian;
        cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
        cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
        return cov;
    }

    inline vect3 SO3ToEuler(const SO3 &orient)
    {
        Eigen::Matrix<double, 3, 1> _ang;
        Eigen::Vector4d q_data = orient.coeffs().transpose();

        double sqw = q_data[3] * q_data[3];
        double sqx = q_data[0] * q_data[0];
        double sqy = q_data[1] * q_data[1];
        double sqz = q_data[2] * q_data[2];
        double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
        double test = q_data[3] * q_data[1] - q_data[2] * q_data[0];

        if (test > 0.49999 * unit)
        { // singularity at north pole

            _ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI / 2, 0;
            double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
            vect3 euler_ang(temp, 3);
            return euler_ang;
        }
        if (test < -0.49999 * unit)
        { // singularity at south pole
            _ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI / 2, 0;
            double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
            vect3 euler_ang(temp, 3);
            return euler_ang;
        }

        _ang << std::atan2(2 * q_data[0] * q_data[3] + 2 * q_data[1] * q_data[2], -sqx - sqy + sqz + sqw),
            std::asin(2 * test / unit),
            std::atan2(2 * q_data[2] * q_data[3] + 2 * q_data[1] * q_data[0], sqx - sqy - sqz + sqw);
        double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
        vect3 euler_ang(temp, 3);

        return euler_ang;
    }

} // namespace g2m_lio

#endif

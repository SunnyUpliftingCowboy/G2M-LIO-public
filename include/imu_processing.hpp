/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#ifndef AKF_LIO_IMU_PROCESSING_H
#define AKF_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <ros/assert.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"
#include "utils.h"

namespace g2m_lio
{

    constexpr int MAX_INI_COUNT = 20;

    bool time_list(const PointType2 &x, const PointType2 &y) { return (x.curvature < y.curvature); };

    /// IMU Process and undistortion
    class ImuProcess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuProcess();
        ~ImuProcess();

        void Reset();
        void SetExtrinsic(const common::V3D &transl, const common::M3D &rot);
        void SetGyrCov(const common::V3D &scaler);
        void SetAccCov(const common::V3D &scaler);
        void SetGyrBiasCov(const common::V3D &b_g);
        void SetAccBiasCov(const common::V3D &b_a);
        void SetConingCompensation(bool enable) { coning_comp_en_ = enable; }
        void SetScullingCompensation(bool enable) { sculling_comp_en_ = enable; }
        void SetGravity(double g) { gravity_m_s2_ = g; }
        void SetIMUAccUnit(double unit) { imu_acc_unit_ = unit; }
        void SetInitFrameCount(int count) { max_init_count_ = std::max(1, count); }
        void Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                     PointCloudType::Ptr pcl_un_);

        // 获取IMU均值（初始化阶段累积的加速度计和陀螺仪均值）
        common::V3D GetAccMean() const { return mean_acc_; }
        common::V3D GetGyrMean() const { return mean_gyr_; }

        // 获取加速度转换因子（原始IMU读数 -> m/s²）
        double GetAccScaleFactor() const { return acc_scale_factor_; }

        // 检查IMU是否已完成初始化
        bool IsInitialized() const { return !imu_need_init_; }

        std::ofstream fout_imu_;
        Eigen::Matrix<double, 12, 12> Q_;
        common::V3D cov_acc_;
        common::V3D cov_gyr_;
        common::V3D cov_acc_scale_;
        common::V3D cov_gyr_scale_;
        common::V3D cov_bias_gyr_;
        common::V3D cov_bias_acc_;

        // 圆锥/划桨补偿开关
        bool coning_comp_en_ = false;
        bool sculling_comp_en_ = false;
        int max_init_count_ = 20;

    private:
        void IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
        void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                          PointCloudType &pcl_out);
        void OnlyPropagate(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                           PointCloudType &pcl_out);

        PointCloudType::Ptr cur_pcl_un_;
        sensor_msgs::ImuConstPtr last_imu_;
        std::deque<sensor_msgs::ImuConstPtr> v_imu_;
        std::vector<common::Pose6D> IMUpose_;
        std::vector<common::M3D> v_rot_pcl_;
        common::M3D Lidar_R_wrt_IMU_;
        common::V3D Lidar_T_wrt_IMU_;
        common::V3D mean_acc_;
        common::V3D mean_gyr_;
        common::V3D angvel_last_;
        common::V3D acc_s_last_;
        double last_lidar_end_time_ = 0;
        int init_iter_num_ = 1;
        bool b_first_frame_ = true;
        bool imu_need_init_ = true;

        // 可配置的重力加速度和IMU加速度计单位
        double gravity_m_s2_ = common::G_m_s2;   // 重力加速度 (m/s²), 默认 9.81
        double imu_acc_unit_ = -1.0;             // IMU在1g下的原始读数, 负值=自动计算
        double acc_scale_factor_ = 1.0;           // 原始IMU读数 -> m/s² 的转换因子
    };

    ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true)
    {
        init_iter_num_ = 1;
        Q_ = process_noise_cov();
        cov_acc_ = common::V3D(0.1, 0.1, 0.1);
        cov_gyr_ = common::V3D(0.1, 0.1, 0.1);
        cov_bias_gyr_ = common::V3D(0.0001, 0.0001, 0.0001);
        cov_bias_acc_ = common::V3D(0.0001, 0.0001, 0.0001);
        mean_acc_ = common::V3D(0, 0, -1.0);
        mean_gyr_ = common::V3D(0, 0, 0);
        angvel_last_ = common::Zero3d;
        acc_s_last_ = common::Zero3d;
        Lidar_T_wrt_IMU_ = common::Zero3d;
        Lidar_R_wrt_IMU_ = common::Eye3d;
        last_imu_.reset(new sensor_msgs::Imu());
    }

    ImuProcess::~ImuProcess() {}

    void ImuProcess::Reset()
    {
        mean_acc_ = common::V3D(0, 0, -1.0);
        mean_gyr_ = common::V3D(0, 0, 0);
        angvel_last_ = common::Zero3d;
        acc_s_last_ = common::Zero3d;
        last_lidar_end_time_ = 0.0;
        imu_need_init_ = true;
        init_iter_num_ = 1;
        v_imu_.clear();
        IMUpose_.clear();
        last_imu_.reset(new sensor_msgs::Imu());
        cur_pcl_un_.reset(new PointCloudType());
    }

    void ImuProcess::SetExtrinsic(const common::V3D &transl, const common::M3D &rot)
    {
        Lidar_T_wrt_IMU_ = transl;
        Lidar_R_wrt_IMU_ = rot;
    }

    void ImuProcess::SetGyrCov(const common::V3D &scaler) { cov_gyr_scale_ = scaler; }

    void ImuProcess::SetAccCov(const common::V3D &scaler) { cov_acc_scale_ = scaler; }

    void ImuProcess::SetGyrBiasCov(const common::V3D &b_g) { cov_bias_gyr_ = b_g; }

    void ImuProcess::SetAccBiasCov(const common::V3D &b_a) { cov_bias_acc_ = b_a; }

    void ImuProcess::IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                             int &N)
    {
        /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity_ **/

        common::V3D cur_acc, cur_gyr;

        if (b_first_frame_)
        {
            Reset();
            N = 1;
            b_first_frame_ = false;
            const auto &imu_acc = meas.imu_.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu_.front()->angular_velocity;
            mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        }

        for (const auto &imu : meas.imu_)
        {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

            mean_acc_ += (cur_acc - mean_acc_) / N;
            mean_gyr_ += (cur_gyr - mean_gyr_) / N;
            // 下面这段代码似乎没有作用？可能需要进一步检查
            cov_acc_ =
                cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            cov_gyr_ =
                cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

            N++;
        }
        state_ikfom init_state = kf_state.get_x();
        init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * gravity_m_s2_);

        init_state.bg = mean_gyr_;
        init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
        init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
        kf_state.change_x(init_state);

        esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
        init_P.setIdentity();
        init_P(0, 0) = init_P(1, 1) = init_P(2, 2) = 0.0001;
        init_P(3, 3) = init_P(4, 4) = init_P(5, 5) = 0.000001;
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
        init_P(21, 21) = init_P(22, 22) = 0.00001;
        kf_state.change_P(init_P);
        last_imu_ = meas.imu_.back();
        // The initialized state is anchored at this frame end, not at the
        // timestamp of its last raw IMU (which can precede the boundary).
        last_lidar_end_time_ = meas.lidar_end_time_;
    }

    void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                  PointCloudType &pcl_out)
    {
        /*** add the imu_ of the last frame-tail to the of current frame-head ***/
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_);
        if (meas.imu_after_end_)
        {
            v_imu.push_back(meas.imu_after_end_);
        }
        const double &pcl_beg_time = meas.lidar_bag_time_;
        const double &pcl_end_time = meas.lidar_end_time_;

        // Validate coverage before changing the filter. Never extrapolate a
        // stale input, or silently propagate past the requested frame end.
        if (pcl_end_time <= last_lidar_end_time_)
        {
            LOG(WARNING) << "[IMU] Stale lidar frame: end=" << pcl_end_time
                         << " state_time=" << last_lidar_end_time_ << "; skip propagation";
            return;
        }
        if (v_imu.front()->header.stamp.toSec() > last_lidar_end_time_ ||
            v_imu.back()->header.stamp.toSec() < pcl_end_time)
        {
            LOG(WARNING) << "[IMU] Missing timestamp coverage: state_time=" << last_lidar_end_time_
                         << " imu_range=[" << v_imu.front()->header.stamp.toSec() << ", "
                         << v_imu.back()->header.stamp.toSec() << "] lidar_end=" << pcl_end_time
                         << "; skip propagation";
            return;
        }
        for (size_t i = 1; i < v_imu.size(); ++i)
        {
            if (v_imu[i]->header.stamp < v_imu[i - 1]->header.stamp)
            {
                LOG(WARNING) << "[IMU] Non-monotonic timestamps; skip scan";
                return;
            }
        }

        /*** sort point clouds by offset time ***/
        pcl_out = *(meas.lidar_);
        sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

        // A spinning scan can start before the previous scan's propagation
        // end. Those old points have no trajectory in this frame; discard
        // them while retaining all points at/after the current state time.
        const double min_valid_offset = last_lidar_end_time_ - pcl_beg_time;
        const auto first_valid = std::lower_bound(
            pcl_out.points.begin(), pcl_out.points.end(), min_valid_offset - 1e-6,
            [](const PointType2 &point, double offset) { return point.curvature / 1000.0 < offset; });
        pcl_out.points.erase(pcl_out.points.begin(), first_valid);

        /*** Initialize IMU pose ***/
        state_ikfom imu_state = kf_state.get_x();
        IMUpose_.clear();
        IMUpose_.push_back(common::set_pose6d(last_lidar_end_time_ - pcl_beg_time,
                                              acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                              imu_state.rot.toRotationMatrix()));

        /*** forward propagation at each imu_ point ***/
        double dt = 0;

        Q_.block<3, 3>(0, 0) = cov_gyr_scale_.asDiagonal();
        Q_.block<3, 3>(3, 3) = cov_acc_scale_.asDiagonal();
        Q_.block<3, 3>(6, 6) = cov_bias_gyr_.asDiagonal();
        Q_.block<3, 3>(9, 9) = cov_bias_acc_.asDiagonal();

        input_ikfom in;
        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
        {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);

            const double head_time = head->header.stamp.toSec();
            const double tail_time = tail->header.stamp.toSec();
            const double start_time = std::max(head_time, last_lidar_end_time_);
            const double end_time = std::min(tail_time, pcl_end_time);
            if (tail_time <= head_time || end_time <= start_time)
            {
                continue;
            }
            dt = end_time - start_time;

            // 读取原始 IMU 测量值（Eigen 向量形式）
            common::V3D w1(head->angular_velocity.x, head->angular_velocity.y, head->angular_velocity.z);
            common::V3D w2(tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z);
            common::V3D a1(head->linear_acceleration.x, head->linear_acceleration.y, head->linear_acceleration.z);
            common::V3D a2(tail->linear_acceleration.x, tail->linear_acceleration.y, tail->linear_acceleration.z);

            // Interpolate BOTH endpoints of the clipped interval. Use the
            // original samples for both expressions (do not overwrite w1/a1
            // before computing the right endpoint).
            const double alpha = (start_time - head_time) / (tail_time - head_time);
            const double beta = (end_time - head_time) / (tail_time - head_time);
            const common::V3D w_start = (1.0 - alpha) * w1 + alpha * w2;
            const common::V3D a_start = (1.0 - alpha) * a1 + alpha * a2;
            w2 = ((1.0 - beta) * w1 + beta * w2).eval();
            a2 = ((1.0 - beta) * a1 + beta * a2).eval();
            w1 = w_start;
            a1 = a_start;

            // Pass rates in physical units. Bias removal and compensation
            // belong to the process model, together with their Jacobians.
            in.gyro = 0.5 * (w1 + w2);
            in.acc = (0.5 * (a1 + a2)) * acc_scale_factor_;
            in.gyro_delta = w2 - w1;
            in.acc_delta = (a2 - a1) * acc_scale_factor_;
            in.integration_dt = dt;
            in.coning_compensation = coning_comp_en_;
            in.sculling_compensation = sculling_comp_en_;
            const state_ikfom state_before_predict = kf_state.get_x();
            const auto corrected = correct_imu_input(state_before_predict, in);
            common::V3D gravity_world;
            for (int i = 0; i < 3; ++i)
            {
                gravity_world[i] = state_before_predict.grav[i];
            }
            const common::V3D acceleration_world =
                state_before_predict.rot * corrected.acc + gravity_world;
            kf_state.predict(dt, Q_, in);
            state_ikfom state_after_predict = kf_state.get_x();
            state_after_predict.pos += 0.5 * acceleration_world * dt * dt;
            kf_state.change_x(state_after_predict);

            /* save the poses at each IMU measurements */
            imu_state = kf_state.get_x();
            angvel_last_ = corrected.gyro;
            // Cache the actual endpoint measurement, not the interval force
            // (which is expressed in the starting body frame).
            acc_s_last_ = imu_state.rot * (a2 * acc_scale_factor_ - imu_state.ba);
            for (int i = 0; i < 3; i++)
            {
                acc_s_last_[i] += imu_state.grav[i];
            }
            const double offs_t = end_time - pcl_beg_time;
            IMUpose_.emplace_back(common::set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                                     imu_state.rot.toRotationMatrix()));
        }

        imu_state = kf_state.get_x();
        // Keep the last original sample at/before the boundary. The borrowed
        // future sample must not replace it; both sides are needed next time.
        if (!meas.imu_.empty())
        {
            last_imu_ = meas.imu_.back();
        }
        last_lidar_end_time_ = pcl_end_time;

        /*** undistort each lidar point (backward propagation) ***/
        if (pcl_out.points.empty())
        {
            return;
        }
        // Interpolate the actual predicted nodes, including the frame-end
        // node. SLERP/linear translation are short-interval approximations,
        // but agree exactly with the filter at every node for all switches.
        // Do not mix a separately integrated p+v*dt+0.5*a*dt^2 trajectory with
        // the filter's first-order position update.
        size_t segment = 0;
        for (auto &point : pcl_out.points)
        {
            const double offset = point.curvature / 1000.0;
            if (offset < IMUpose_.front().offset_time - 1e-6 ||
                offset > IMUpose_.back().offset_time + 1e-6)
            {
                LOG(WARNING) << "[IMU] Point timestamp outside predicted trajectory: point_offset=" << offset
                             << " trajectory_offsets=[" << IMUpose_.front().offset_time << ", "
                             << IMUpose_.back().offset_time << "]; discard scan";
                pcl_out.clear();
                return;
            }
            while (segment + 2 < IMUpose_.size() && offset > IMUpose_[segment + 1].offset_time)
            {
                ++segment;
            }
            const auto &head = IMUpose_[segment];
            const auto &tail = IMUpose_[segment + 1];
            const double ratio = std::clamp((offset - head.offset_time) /
                                              (tail.offset_time - head.offset_time), 0.0, 1.0);
            const Eigen::Quaterniond q_head(common::MatFromArray(head.rot));
            const Eigen::Quaterniond q_tail(common::MatFromArray(tail.rot));
            const common::M3D R_i = q_head.slerp(ratio, q_tail).toRotationMatrix();
            const common::V3D pos_i = (1.0 - ratio) * common::VecFromArray(head.pos) +
                                      ratio * common::VecFromArray(tail.pos);
            const common::V3D T_ei = pos_i - imu_state.pos;
            const common::V3D P_i(point.x, point.y, point.z);
            const common::V3D p_compensate =
                imu_state.offset_R_L_I.conjugate() *
                (imu_state.rot.conjugate() *
                     (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                 imu_state.offset_T_L_I);
            point.x = p_compensate.x();
            point.y = p_compensate.y();
            point.z = p_compensate.z();
        }
    }

    void ImuProcess::Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                             PointCloudType::Ptr cur_pcl_un_)
    {
        // Never leave a previous scan in the output on an initialization or
        // coverage failure. A frame with no new IMUs may still be bracketed
        // by last_imu_ and imu_after_end_.
        cur_pcl_un_->clear();
        if (meas.imu_.empty() && (imu_need_init_ || !meas.imu_after_end_))
        {
            return;
        }

        ROS_ASSERT(meas.lidar_ != nullptr);

        if (imu_need_init_)
        {
            /// The very first lidar frame
            IMUInit(meas, kf_state, init_iter_num_);

            imu_need_init_ = true;

            last_imu_ = meas.imu_.back();

            state_ikfom imu_state = kf_state.get_x();
            if (init_iter_num_ > max_init_count_)
            {
                // --- 加速度转换因子计算与验证 ---
                // 默认使用IMU初始化过程中计算出的 mean_acc_.norm() 作为转换基准
                double computed_unit = mean_acc_.norm();
                double chosen_unit = computed_unit;

                if (imu_acc_unit_ > 0.0)
                {
                    // 用户通过配置文件指定了IMU加速度计单位
                    double ratio = imu_acc_unit_ / computed_unit;
                    if (ratio > 2.0 || ratio < 0.5)
                    {
                        LOG(WARNING) << "[IMU Init] Configured imu_acc_unit (" << std::fixed
                                     << std::setprecision(2) << imu_acc_unit_
                                     << ") differs significantly from computed value ("
                                     << computed_unit << "), ratio=" << ratio
                                     << ". Using auto-computed value instead. "
                                     << "Please check your YAML config parameter 'mapping/imu_acc_unit'.";
                        chosen_unit = computed_unit;
                    }
                    else
                    {
                        chosen_unit = imu_acc_unit_;
                        LOG(INFO) << "[IMU Init] Using configured imu_acc_unit=" << chosen_unit
                                  << " (computed=" << computed_unit << ", ratio=" << ratio << ")";
                    }
                }

                acc_scale_factor_ = gravity_m_s2_ / chosen_unit;

                LOG(INFO) << "[IMU Init] gravity=" << gravity_m_s2_ << " m/s², "
                          << "acc_unit_raw=" << chosen_unit << ", "
                          << "acc_scale=" << acc_scale_factor_;

                imu_need_init_ = false;

                LOG(INFO) << "[IMU Init] Done. acc_norm=" << std::fixed << std::setprecision(6)
                          << mean_acc_.norm() << " raw, grav=" << kf_state.get_x().grav
                          << ", bg=" << kf_state.get_x().bg.transpose();

                cov_acc_ = cov_acc_scale_;
                cov_gyr_ = cov_gyr_scale_;
                LOG(INFO) << "IMU Initial Done";
                fout_imu_.open(common::DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
            }

            return;
        }

        Timer::Evaluate(
            [&, this]()
            {
                UndistortPcl(meas, kf_state, *cur_pcl_un_);
            },
            "Undistort Pcl");
    }
} // namespace g2m_lio

#endif

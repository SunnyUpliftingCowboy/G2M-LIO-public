/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <execution>
#include <fstream>
#include <chrono>
#include <sstream>
#include <unordered_set>
#include <iomanip>
#include <limits>

#include "laser_mapping.h"
#include "utils.h"

#include <tbb/parallel_for.h>

#include <opencv2/opencv.hpp>

namespace g2m_lio
{

    bool LaserMapping::InitROS(ros::NodeHandle &nh)
    {
        LoadParams(nh);
        SubAndPubToROS(nh);

        // localmap init (after LoadParams)
        ivox_ = std::make_shared<IVoxType>(ivox_options_);

        // esekf init
        std::vector<double> epsi(23, 0.001);
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
            { ObsModel(s, ekfom_data); },
            options::NUM_MAX_ITERATIONS, epsi.data());

        return true;
    }

    bool LaserMapping::LoadParams(ros::NodeHandle &nh)
    {
        // get params from param server
        int lidar_type, ivox_nearby_type;
        double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
        common::V3D lidar_T_wrt_IMU;
        common::M3D lidar_R_wrt_IMU;

        nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
        nh.param<double>("preprocess/max_range", preprocess_->Max_range(), 500.0);
        nh.param<bool>("preprocess/enable_max_range_filter", preprocess_->EnableMaxRangeFilter(), true);
        LOG(INFO) << "max range filter " << (preprocess_->EnableMaxRangeFilter() ? "enabled" : "disabled")
                  << ", limit=" << preprocess_->Max_range() << " m";
        nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-6);
        nh.param<int>("preprocess/lidar_type", lidar_type, 1);
        nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);

        nh.param<float>("lio/lidar_cov", options::LIDAR_COV, 0.001);
        nh.param<double>("lio/lidar_range_sigma", lidar_range_sigma_, 0.0);
        nh.param<double>("lio/lidar_angle_sigma_deg", lidar_angle_sigma_deg_, 0.0);
        nh.param<double>("lio/residual_gate_sigma", residual_gate_sigma_, 3.0);
        nh.param<double>("lio/residual_variance_floor", residual_variance_floor_, 0.001);
        nh.param<int>("lio/imu_init_frame_count", imu_init_frame_count_, 20);
        p_imu_->SetInitFrameCount(imu_init_frame_count_);
        nh.param<double>("lio/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("lio/acc_cov", acc_cov, 0.1);
        nh.param<double>("lio/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("lio/b_acc_cov", b_acc_cov, 0.0001);
        nh.param<double>("lio/gravity_m_s2", gravity_m_s2_, 9.81);
        nh.param<double>("lio/imu_acc_unit", imu_acc_unit_, -1.0);
        nh.param<bool>("lio/coning_compensation", imu_coning_comp_en_, false);
        nh.param<bool>("lio/sculling_compensation", imu_sculling_comp_en_, false);
        nh.param<std::vector<double>>("lio/extrinsic_T", extrinT_, std::vector<double>());
        nh.param<std::vector<double>>("lio/extrinsic_R", extrinR_, std::vector<double>());

        nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
        nh.param<bool>("publish/scan_reg_pub_en", scan_reg_pub_en_, false);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
        nh.param<bool>("publish/map_publish_en", map_pub_en_, false);
        nh.param<bool>("publish/voxel_input_pub_en", voxel_input_pub_en_, false);
        nh.param<bool>("publish/gaussian_publish_en", gaussian_publish_en_, false);
        nh.param<double>("publish/gaussian_pub_dis", gaussian_pub_dis_, 10);
        nh.param<double>("publish/gaussian_pub_min_cnt", gaussian_pub_min_cnt_, 5);
        nh.param<bool>("publish/obs_error_viz_en", obs_error_viz_en_, false);
        nh.param<bool>("publish/voxel_debug_en", voxel_debug_en_, false);
        nh.param<bool>("publish/voxel_size_monitor_en", voxel_size_monitor_en_, false);

        nh.param<bool>("voxel_map/adap_voxel_size_en", adap_voxel_size_en_, true);
        nh.param<bool>("voxel_map/pcl_voxel_downsample_en", pcl_voxel_downsample_en_, false);
        nh.param<bool>("gravity_align_en", gravity_align_en_, false);
        nh.param<int>("voxel_map/target_point_size", target_point_size_, 2000);
        nh.param<int>("voxel_map/max_gaussian_points", options::MAX_FEA_NUM, 10000);
        nh.param<int>("voxel_map/min_support_voxels", min_support_voxels_, 2);
        nh.param<int>("voxel_map/max_support_voxels", max_support_voxels_, 7);
        nh.param<bool>("voxel_map/support_voxel_use_map_resolution",
                       support_voxel_use_map_resolution_, false);
        min_support_voxels_ = std::clamp(min_support_voxels_, 1, options::NUM_MATCH_POINTS);
        max_support_voxels_ = std::clamp(max_support_voxels_, min_support_voxels_, options::NUM_MATCH_POINTS);
        nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
        nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
        nh.param<double>("voxel_map/ivox_grid_resolution", ivox_options_.resolution_, 0.5);
        nh.param<double>("voxel_map/scan_voxel_resolution", scan_voxel_resolution_, 0.5);
        ivox_eigen_threshold_ = 4.0 / 25.0 * ivox_options_.resolution_ * ivox_options_.resolution_;

        nh.param<int>("voxel_map/ivox_nearby_type", ivox_nearby_type, 26);
        nh.param<bool>("voxel_map/cross_voxel_merge_en", cross_voxel_merge_en_, true);
        nh.param<float>("t_mal", options::T_MAL, 11.28);
        nh.param<double>("t_stop_pseudo_merge", t_stop_pseudo_merge_, 11.28);
        nh.param<double>("time_to_delete_local_map", options::TIME_TO_DELETE_LOCAL_MAP, 1000);

        // 点点ICP参数


        nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
        nh.param<bool>("mapping_log_enable", mapping_log_en_, false);
        nh.param<bool>("consistency_log_enable", consistency_log_en_, false);
        nh.param<std::string>("consistency_file_path", consistency_file_path_,
                              std::string(ROOT_DIR) + "Log/consistency.csv");
        kf_.set_update_diagnostics_enabled(consistency_log_en_);

        LOG(INFO) << "lidar_type " << lidar_type;
        if (lidar_type == 1)
        {
            preprocess_->SetLidarType(LidarType::AVIA);
            LOG(INFO) << "Using AVIA Lidar";
        }
        else if (lidar_type == 2)
        {
            preprocess_->SetLidarType(LidarType::VELO32);
            LOG(INFO) << "Using Velodyne 32 Lidar";
        }
        else if (lidar_type == 3)
        {
            preprocess_->SetLidarType(LidarType::OUST64);
            LOG(INFO) << "Using OUST 64 Lidar";
        }
        else
        {
            LOG(WARNING) << "unknown lidar_type";
            return false;
        }

        if (ivox_nearby_type == 0)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        }
        else if (ivox_nearby_type == 6)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        }
        else if (ivox_nearby_type == 18)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }
        else if (ivox_nearby_type == 26)
        {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        }
        else
        {
            LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }

        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";
        gt_path_.header.stamp = ros::Time::now();
        gt_path_.header.frame_id = "camera_init";

        fout_pre.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_pre.txt")), std::ios::out);
        fout_out.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_out.txt")), std::ios::out);
        fout_cov.open((std::string(std::string(ROOT_DIR) + "Log/" + "mat_cov.txt")), std::ios::out);
        if (consistency_log_en_)
        {
            fout_consistency_.open(consistency_file_path_, std::ios::out);
            fout_consistency_ << "timestamp,x,y,z,qx,qy,qz,qw";
            for (int row = 0; row < 6; ++row)
                for (int col = 0; col < 6; ++col)
                    fout_consistency_ << ",P" << row << col;
            fout_consistency_ << ",nis,nis_dof,nis_per_dof,whitened_residual_sum\n";
        }
        if (fout_pre && fout_out && fout_cov)
            std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
        else
            std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

        lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
        lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

        p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
        p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        p_imu_->SetGravity(gravity_m_s2_);
        p_imu_->SetIMUAccUnit(imu_acc_unit_);
        p_imu_->SetConingCompensation(imu_coning_comp_en_);
        p_imu_->SetScullingCompensation(imu_sculling_comp_en_);

        // 地图点协方差传播参数
        nh.param<bool>("map_covariance_propagation/enabled", map_cov_prop_enabled_, true);
        nh.param<bool>("map_covariance_propagation/covariance_propagation/enabled", cov_prop_enabled_, true);
        nh.param<bool>("map_covariance_propagation/visualization/enabled", viz_enabled_, true);
        nh.param<double>("map_covariance_propagation/visualization/max_cov_trace", viz_max_cov_trace_, 0.1);
        nh.param<bool>("map_covariance_propagation/debug/publish_log", debug_log_, true);
        nh.param<double>("common/time_diff_lidar_to_imu", time_diff_lidar_to_imu_, 0.0);

        return true;
    }

    void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh)
    {
        // ROS subscribe initialization
        std::string lidar_topic, imu_topic;
        nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
        nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");

        if (preprocess_->GetLidarType() == LidarType::AVIA)
        {
            sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
                lidar_topic, 200000, [this](const livox_ros_driver::CustomMsg::ConstPtr &msg)
                { LivoxPCLCallBack(msg); });
        }
        else
        {
            sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
                lidar_topic, 200000, [this](const sensor_msgs::PointCloud2::ConstPtr &msg)
                { StandardPCLCallBack(msg); });
        }

        sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                                  [this](const sensor_msgs::Imu::ConstPtr &msg)
                                                  { IMUCallBack(msg); });

        // ROS publisher init
        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";

        pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_dense_world", 100000);
        pub_laser_cloud_reg_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_reg_world", 100000);
        pub_voxel_input_ = nh.advertise<sensor_msgs::PointCloud2>("/voxel_map_input", 100000);
        pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
        pub_path_ = nh.advertise<nav_msgs::Path>("/path", 100000);
        gt_pub_path_ = nh.advertise<nav_msgs::Path>("/gt_path", 100000);
        point_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/scan_gaussian_reg", 10000);
        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
        map_cov_pub = nh.advertise<visualization_msgs::MarkerArray>("/map_gaussian", 10000);
        matched_gaussian_pub = nh.advertise<visualization_msgs::MarkerArray>("/matched_gaussian", 10000);
        voxel_debug_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxel_debug", 10000);
        if (voxel_size_monitor_en_)
            voxel_monitor_pub_ = nh.advertise<std_msgs::Float64MultiArray>("/voxel_size_monitor", 10000);
    }

    LaserMapping::LaserMapping()
    {
        preprocess_.reset(new PointCloudPreprocess());
        p_imu_.reset(new ImuProcess());
    }

    PointType LaserMapping::Merge2(const PointType &pt1, const PointType &pt2)
    {
        // 基元已饱和（点数达到上限），不再融合，直接返回已有基元
        if (pt2.pt_num >= options::MAX_FEA_NUM)
            return pt2;
        if (pt1.pt_num >= options::MAX_FEA_NUM)
            return pt1;

        PointType pt3 = pt2;
        common::V3D p1(pt1.x, pt1.y, pt1.z), p2(pt2.x, pt2.y, pt2.z), p3;
        double ratio_1 = (double)pt1.pt_num / (double)(pt1.pt_num + pt2.pt_num);
        double ratio_2 = (double)pt2.pt_num / (double)(pt1.pt_num + pt2.pt_num);

        p3 = p1 * ratio_1 + p2 * ratio_2;
        pt3.x = p3(0);
        pt3.y = p3(1);
        pt3.z = p3(2);
        pt3.intensity = (pt1.intensity * ratio_1 + pt2.intensity * ratio_2);

        // 分离的协方差融合
        // 位置不确定度融合（加权平均）
        // pt3.cov_pos = ratio_1 * pt1.cov_pos + ratio_2 * pt2.cov_pos;
        pt3.cov_pos = ratio_1 * ratio_1 * pt1.cov_pos + ratio_2 * ratio_2 * pt2.cov_pos;
        // 几何协方差融合（包含相对于新质心的偏移）
        common::M3D offset_cov1 = (p1 - p3) * (p1 - p3).transpose();
        common::M3D offset_cov2 = (p2 - p3) * (p2 - p3).transpose();
        // pt3.cov_geo = ratio_1 * (pt1.cov_geo + offset_cov1) + ratio_2 * (pt2.cov_geo + offset_cov2);
        double ratio_geo_1 = ((double)pt1.pt_num - 1) / (double)(pt1.pt_num + pt2.pt_num - 1);
        double ratio_geo_2 = ((double)pt2.pt_num - 1) / (double)(pt1.pt_num + pt2.pt_num - 1);
        double ratio_geo_3 = (double)pt1.pt_num / (double)(pt1.pt_num + pt2.pt_num - 1);
        double ratio_geo_4 = (double)pt2.pt_num / (double)(pt1.pt_num + pt2.pt_num - 1);
        pt3.cov_geo = ratio_geo_1 * pt1.cov_geo + ratio_geo_2 * pt2.cov_geo + ratio_geo_3 * offset_cov1 + ratio_geo_4 * offset_cov2;

        // Keep the stored count bounded.  Merge2 may be called directly by the
        // map update path, so the invariant must be enforced at the merge point.
        pt3.pt_num = std::min(pt1.pt_num + pt2.pt_num, options::MAX_FEA_NUM);

        pt3.time = pt1.time > pt2.time ? pt1.time : pt2.time;

        return pt3;
    }

    // 将SLAM估计的世界系对齐到"重力朝下"的世界系
    static void alignGravityToWorld(const Eigen::Vector3d& estimated_gravity,
                                    Eigen::Vector3d& position,
                                    Eigen::Matrix3d& rotation,
                                    Eigen::Vector3d& velocity,
                                    Eigen::Vector3d& gravity)
    {
        // 1. 计算旋转：将估计重力转到 (0, 0, -1)
        Eigen::Vector3d target_gravity(0.0, 0.0, -1.0);
        Eigen::Quaterniond q_align =
            Eigen::Quaterniond::FromTwoVectors(estimated_gravity.normalized(),
                                               target_gravity);

        Eigen::Matrix3d R_align = q_align.toRotationMatrix();

        // 2. 统一旋转所有世界系状态量
        position    = R_align * position;
        rotation    = R_align * rotation;
        velocity    = R_align * velocity;
        gravity     = R_align * gravity;
    }

    void LaserMapping::Run()
    {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        if (!SyncPackages())
        {
            return;
        }

        // Core timing starts after synchronization.  It intentionally covers
        // prediction/undistortion, downsampling, IEKF update and incremental
        // map maintenance, but stops before any ROS/debug publication below.
        auto core_frame_start_time = std::chrono::high_resolution_clock::now();

        if (first_lidar_time_ < 1e-6)
            first_lidar_time_ = measures_.lidar_bag_time_;
        if (runtime_pos_log_)
            LOG(INFO) << "measures_.lidar_bag_time_ - first_lidar_time_ " << measures_.lidar_bag_time_ - first_lidar_time_;
        common::V3D euler_cur, ext_euler, print_status;
        ext_euler.setZero();
        print_status.setZero();

        /// IMU process, kf prediction, undistortion
        p_imu_->Process(measures_, kf_, scan_undistort_);
        pred_cov_ = kf_.get_P().block<6, 6>(0, 0);

        if (scan_undistort_->empty() || (scan_undistort_ == nullptr))
        {
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            LOG(WARNING) << "No point, skip this scan!";
            return;
        }

        flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

        /// downsample
        if (pcl_voxel_downsample_en_)
        {
            Timer::Evaluate([&, this]()
                            { PclVoxelGridDownsample(scan_undistort_, scan_down_reg_); },
                            "Downsample PointCloud (PCL VoxelGrid)");
        }
        else
        {
            Timer::Evaluate([&, this]()
                            { VoxelGridDownsample(scan_undistort_, scan_down_reg_); },
                            "Downsample PointCloud");
        }

        if (flg_first_scan_)
        {
            PointVector points_to_add;

            state_point_ = kf_.get_x();
            state_cov_.setZero();

            // --- 初始重力对齐（一次性） ---
            if (gravity_align_en_ && !gravity_aligned_)
            {
                Eigen::Vector3d grav_vec = state_point_.grav.get_vect();
                Eigen::Vector3d grav_old = grav_vec;  // 保存对齐前的完整重力向量 (含 norm)
                Eigen::Vector3d pos = state_point_.pos;
                Eigen::Matrix3d rot = state_point_.rot.toRotationMatrix();
                Eigen::Vector3d vel = state_point_.vel;

                alignGravityToWorld(grav_vec, pos, rot, vel, grav_vec);

                state_point_.pos = pos;
                state_point_.rot = MTK::SO3<double>(rot);
                state_point_.vel = vel;
                state_point_.grav = S2(grav_vec);
                kf_.change_x(state_point_);

                // 协方差同步变换: P_new = T * P * T^T
                Eigen::Matrix3d R_align = Eigen::Quaterniond::FromTwoVectors(
                    grav_old.normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).toRotationMatrix();

                // 计算 S2 重力块的切空间变换矩阵 T_grav (2×2)
                // S2_Bx 给出 3×2 切空间基, T_grav = B_new^T * R * B_old
                S2 s2_old(grav_old);
                S2 s2_new(grav_vec);
                Eigen::Matrix<double, 3, 2> B_old, B_new;
                s2_old.S2_Bx(B_old);
                s2_new.S2_Bx(B_new);
                Eigen::Matrix2d T_grav = B_new.transpose() * R_align * B_old;

                auto P = kf_.get_P();
                Eigen::Matrix<double, 23, 23> T = Eigen::Matrix<double, 23, 23>::Identity();
                T.block<3, 3>(0, 0) = R_align;     // pos (world)
                T.block<3, 3>(3, 3) = R_align;     // rot (world, so(3) ≅ R³)
                T.block<3, 3>(12, 12) = R_align;   // vel (world)
                T.block<2, 2>(21, 21) = T_grav;    // grav (S² 切空间)
                // offset_R_L_I(6-8), offset_T_L_I(9-11), bg(15-17), ba(18-20): body-frame
                P = T * P * T.transpose();
                kf_.change_P(P);

                gravity_aligned_ = true;
                LOG(INFO) << "[Gravity Align] Initial gravity aligned to world (0,0,-1). "
                          << "R_align applied to pos, rot, vel, grav and covariance P.";
            }

            int cur_pts = cur_all_points.size();
            points_to_add.reserve(cur_pts);
            PointType point_world;
            for (int i = 0; i < cur_pts; i++)
            {
                PointBodyToWorld(&(cur_all_points.at(i)), &point_world);
                points_to_add.emplace_back(point_world);
            }
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            if (dense_pub_en_)
                PublishFrameWorld();

            ivox_->AddPoints(points_to_add);
            first_lidar_time_ = measures_.lidar_bag_time_;
            flg_first_scan_ = false;
            return;
        }

        size_t cur_pts = scan_down_reg_.size();
        if (cur_pts < 5)
        {
            PublishOdometry(pub_odom_aft_mapped_);
            if (path_pub_en_)
                PublishPath(pub_path_);
            LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_reg_.size();
            return;
        }
        scan_down_world_.resize(cur_pts);
        if (nearest_points_.size() < cur_pts) nearest_points_.resize(cur_pts);
        for (size_t i = 0; i < cur_pts; ++i) {
            nearest_points_[i].clear();
            if (nearest_points_[i].capacity() < options::NUM_MATCH_POINTS)
                nearest_points_[i].reserve(options::NUM_MATCH_POINTS);
        }
        residuals_.resize(cur_pts);
        residuals_.assign(cur_pts, 0);
        point_selected_surf_.resize(cur_pts);
        plane_coef_.clear();
        plane_coef_.resize(cur_pts);

        state_point_ = kf_.get_x();
        euler_cur = SO3ToEuler(state_point_.rot);

        print_status(0) = avg_voxel_size_;
        print_status(1) = double(cur_pts) / double(target_point_size_);

        ext_euler(0) = kf_.G_cur(3);
        ext_euler(1) = kf_.G_cur(4);
        ext_euler(2) = kf_.G_cur(5);

        fout_pre << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                 << state_point_.pos.transpose() << " " << ext_euler.transpose()
                 << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                 << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;

        // ICP and iterated Kalman filter update
        Timer::Evaluate(
            [&, this]()
            {
                // iterated state estimation
                double solve_H_time = 0;
                // update the observation model, will call nn and point-to-plane residual computation
                kf_.update_iterated_dyn_share_akf(solve_H_time);
                // save the state
                state_point_ = kf_.get_x();
                state_cov_ = kf_.get_P().block<6, 6>(0, 0);
                WriteConsistencyLog();
                // 保存 P 阵对角线（23维状态协方差）
                {
                    Eigen::Matrix<double, 23, 1> p_diag = kf_.get_P().diagonal();
                    fout_cov << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_;
                    for (int i = 0; i < 23; i++)
                        fout_cov << " " << std::setprecision(10) << p_diag(i);
                    fout_cov << std::endl;
                }
                euler_cur_ = SO3ToEuler(state_point_.rot);
                pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

                euler_cur = SO3ToEuler(state_point_.rot);
                ext_euler(0) = kf_.G_cur(0);
                ext_euler(1) = kf_.G_cur(1);
                ext_euler(2) = kf_.G_cur(2);
            },
            "IEKF Solve and Update");

        Timer::Evaluate([&, this]()
                        { MapIncremental(); }, "    Incremental Mapping");

        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(frame_end_time - frame_start_time).count() * 1000;
        Timer::RecordTime("Frame Total", frame_time);

        auto core_frame_end_time = std::chrono::high_resolution_clock::now();
        auto core_frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                   core_frame_end_time - core_frame_start_time)
                                   .count() * 1000;
        Timer::RecordTime("Core Frame Total", core_frame_time);

        if (mapping_log_en_)
            LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
                      << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_
                      << "[Frame Time] Frame " << frame_num_ << " processing time: " << frame_time << " ms";

        // 体素大小自适应监控
        if (voxel_size_monitor_en_)
        {
            std_msgs::Float64MultiArray msg;
            msg.data.resize(6);
            msg.data[0] = static_cast<double>(voxel_monitor_cloud_in_size_);  // 原始输入点云大小
            msg.data[1] = static_cast<double>(target_point_size_);            // 目标降采样点数
            msg.data[2] = static_cast<double>(cur_pts);                       // 实际降采样输出点数
            msg.data[3] = scan_voxel_size_;                                   // 实际降采样体素大小
            msg.data[4] = static_cast<double>(effect_feat_num_);              // 实际参与更新点数
            msg.data[5] = static_cast<double>(cur_pts);                       // 实际更新入地图点数
            voxel_monitor_pub_.publish(msg);
        }

        PublishOdometry(pub_odom_aft_mapped_);

        if (path_pub_en_)
            PublishPath(pub_path_);

        if (dense_pub_en_)
            PublishFrameWorld();

        if (scan_reg_pub_en_)
            PublishFrameRegWorld(pub_laser_cloud_reg_world_);

        if (map_pub_en_ && frame_num_ % 10 == 0)
        {
            PublishMap();
        }

        if (voxel_debug_en_ && frame_num_ % 20 == 0)
        {
            PublishVoxelDebug();
        }

        // 独立发布当前帧参与配准的高斯基元（不受 map_publish_en 控制）
        if (gaussian_publish_en_)
        {
            PublishMatchedGaussian();
        }

        print_status(2) = double(effect_feat_num_) / double(cur_pts);

        fout_out << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << euler_cur.transpose() << " "
                 << state_point_.pos.transpose() << " " << ext_euler.transpose()
                 << " " << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                 << state_point_.ba.transpose() << " " << state_point_.grav << " " << print_status.transpose() << std::endl;

        frame_num_++;
    }

    void LaserMapping::UpdateScanVoxelSize(const PointCloudType::Ptr &cloud_in)
    {
        const size_t cloud_in_size = cloud_in->size();
        std::vector<size_t> index0;
        if (adap_voxel_size_en_ && target_point_size_ < cloud_in_size)
        {
            index0.resize(cloud_in_size);
            for (size_t i = 0; i < cloud_in_size; ++i)
            {
                index0[i] = i;
            }
        }

        voxel_monitor_cloud_in_size_ = cloud_in_size;

        if (adap_voxel_size_en_ && target_point_size_ >= cloud_in_size)
        {
            scan_voxel_size_ = min_voxel_size_;
        }
        else
        {
            if (adap_voxel_size_en_)
            {
                int max_sample_cnt = 10;

                if (!voxel_size_sliding_.empty())
                {
                    scan_voxel_size_ = 0;
                    for (auto &it : voxel_size_sliding_)
                    {
                        scan_voxel_size_ += it;
                    }
                    scan_voxel_size_ /= voxel_size_sliding_.size();
                    scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f;
                }

                for (int iter = 0; iter < max_sample_cnt; iter++)
                {
                    cur_iter = iter;
                    std::unordered_map<size_t, size_t> sample_volume_hash;
                    sample_volume_hash.clear();
                    sample_volume_hash.reserve(cloud_in_size / 2);

                    std::vector<size_t> hash_values(cloud_in_size);
                    std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                                  {
                    common::V3D p3d_local;
                    p3d_local << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
                    for (int j = 0; j < 3; j++) {
                        p3d_local[j] = p3d_local[j] / scan_voxel_size_;
                    }
                    int64_t ijk0_local = int64_t(floor(p3d_local[0]));
                    int64_t ijk1_local = int64_t(floor(p3d_local[1]));
                    int64_t ijk2_local = int64_t(floor(p3d_local[2]));
                    hash_values[i] = ComputeVoxelHash(ijk0_local, ijk1_local, ijk2_local); });

                    for (size_t i = 0; i < cloud_in_size; ++i)
                    {
                        sample_volume_hash[hash_values[i]]++;
                    }

                    double sample_target_ratio = double(sample_volume_hash.size()) / double(target_point_size_);

                    if (fabs(sample_target_ratio - 1.0f) < 0.1f)
                    {
                        break;
                    }
                    else
                    {
                        double last_scan_voxel_size_ = scan_voxel_size_;
                        scan_voxel_size_ = (sample_target_ratio / 2.0f + 0.5f) * scan_voxel_size_;
                        scan_voxel_size_ = round(scan_voxel_size_ / 0.001f) * 0.001f;
                        if (fabs(last_scan_voxel_size_ - scan_voxel_size_) < 0.001f)
                            break;
                    }
                }
                avg_voxel_size_ = scan_voxel_size_;
                voxel_size_sliding_.push_back(scan_voxel_size_);
                if (voxel_size_sliding_.size() > 3)
                    voxel_size_sliding_.pop_front();
            }
            else
            {
                scan_voxel_size_ = scan_voxel_resolution_;
            }
        }

        if (scan_voxel_size_ < min_voxel_size_)
            scan_voxel_size_ = min_voxel_size_;
    }

    void LaserMapping::VoxelGridDownsample(PointCloudType::Ptr &cloud_in, PointVector &cloud_down_reg)
    {
        size_t cloud_in_size = cloud_in->size();

        cloud_down_reg.clear();
        cloud_down_reg.reserve(cloud_in_size);

        cur_all_points.clear();
        cur_all_points.resize(cloud_in_size);

        std::vector<size_t> index0(cloud_in_size);

        for (size_t i = 0; i < cloud_in_size; ++i)
        {
            index0[i] = i;
        }
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
                          PointType &tp = cur_all_points[i];
                          common::V3D pt3d;
                          common::M3D body_cov;
                          tp.x = cloud_in->points[i].x;
                          tp.y = cloud_in->points[i].y;
                          tp.z = cloud_in->points[i].z;

                          tp.intensity = cloud_in->points[i].intensity;
                          tp.pt_num = 1;
                          tp.time = measures_.lidar_bag_time_ - first_lidar_time_ + cloud_in->points[i].curvature / 1000;
                          pt3d << cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z;
                          // 初始化协方差：分离位置不确定度和几何分布
                          tp.cov_pos = CovariancePropagation::ComputeLidarMeasurementCovariance(pt3d, lidar_range_sigma_, lidar_angle_sigma_deg_) + options::LIDAR_COV * Eigen::Matrix3d::Identity();
                          tp.cov_geo = Eigen::Matrix3d::Zero();                          // 单点无几何分布
                      });

        UpdateScanVoxelSize(cloud_in);

        std::unordered_map<size_t, PointVector> leaves_;
        leaves_.clear();
        leaves_.reserve(cloud_in_size / 2);

        std::vector<size_t> final_hash_values(cloud_in_size);
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
        common::V3D loc_xyz_local;
        loc_xyz_local << cur_all_points[i].x, cur_all_points[i].y, cur_all_points[i].z;
        for (int j = 0; j < 3; j++) {
            loc_xyz_local[j] = loc_xyz_local[j] / scan_voxel_size_;
        }
        int64_t ijk0_local = int64_t(floor(loc_xyz_local[0]));
        int64_t ijk1_local = int64_t(floor(loc_xyz_local[1]));
        int64_t ijk2_local = int64_t(floor(loc_xyz_local[2]));
        final_hash_values[i] = ComputeVoxelHash(ijk0_local, ijk1_local, ijk2_local); });

        for (size_t cp = 0; cp < cloud_in_size; cp++)
        {
            auto &voxel = leaves_[final_hash_values[cp]];
            if (voxel.empty())
            {
                voxel.reserve(4);
            }
            voxel.push_back(cur_all_points[cp]);
        }

        cloud_down_reg.reserve(leaves_.size());
        for (const auto &leaf : leaves_)
        {
            cloud_down_reg.push_back(leaf.second.front());
        }
        return;
    }

    void LaserMapping::PclVoxelGridDownsample(PointCloudType::Ptr &cloud_in, PointVector &cloud_down_reg)
    {
        UpdateScanVoxelSize(cloud_in);

        // PCL VoxelGrid selects the centroid of each occupied voxel.
        pcl::VoxelGrid<pcl::PointXYZINormal> filter;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pcl_input(new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl_input->reserve(cloud_in->size());
        for (const auto &p : cloud_in->points) {
            pcl::PointXYZINormal q;
            q.x = p.x; q.y = p.y; q.z = p.z; q.intensity = p.intensity;
            pcl_input->push_back(q);
        }
        filter.setInputCloud(pcl_input);
        const float leaf = static_cast<float>(scan_voxel_size_);
        filter.setLeafSize(leaf, leaf, leaf);
        pcl::PointCloud<pcl::PointXYZINormal> filtered;
        filter.filter(filtered);

        cloud_down_reg.clear();
        cloud_down_reg.reserve(filtered.size());
        for (const auto &p : filtered.points)
        {
            PointType out;
            out.x = p.x; out.y = p.y; out.z = p.z; out.intensity = p.intensity;
            out.pt_num = 1;
            out.time = measures_.lidar_bag_time_ - first_lidar_time_;
            out.cov_pos = CovariancePropagation::ComputeLidarMeasurementCovariance(Eigen::Vector3d(out.x, out.y, out.z), lidar_range_sigma_, lidar_angle_sigma_deg_) + options::LIDAR_COV * Eigen::Matrix3d::Identity();
            out.cov_geo = Eigen::Matrix3d::Zero();
            cloud_down_reg.push_back(out);
        }
    }

    void LaserMapping::VoxelGridDownsampleMerge(PointCloudType::Ptr &cloud_in, PointVector &cloud_down_reg)
    {
        size_t cloud_in_size = cloud_in->size();

        cloud_down_reg.clear();
        cloud_down_reg.reserve(cloud_in_size);

        cur_all_points.clear();
        cur_all_points.resize(cloud_in_size);

        std::vector<size_t> index0(cloud_in_size);

        for (size_t i = 0; i < cloud_in_size; ++i)
        {
            index0[i] = i;
        }
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
                          PointType &tp = cur_all_points[i];
                          tp.x = cloud_in->points[i].x;
                          tp.y = cloud_in->points[i].y;
                          tp.z = cloud_in->points[i].z;
                          tp.intensity = cloud_in->points[i].intensity;
                          tp.pt_num = 1;
                          tp.time = measures_.lidar_bag_time_ - first_lidar_time_ + cloud_in->points[i].curvature / 1000;
                          // 初始化协方差：分离位置不确定度和几何分布
                          tp.cov_pos = CovariancePropagation::ComputeLidarMeasurementCovariance(Eigen::Vector3d(tp.x, tp.y, tp.z), lidar_range_sigma_, lidar_angle_sigma_deg_) + options::LIDAR_COV * Eigen::Matrix3d::Identity();
                          tp.cov_geo = Eigen::Matrix3d::Zero();                          // 单点无几何分布
                      });

        UpdateScanVoxelSize(cloud_in);

        // --- 体素哈希分配（与原函数完全相同）---
        std::unordered_map<size_t, PointVector> leaves_;
        leaves_.clear();
        leaves_.reserve(cloud_in_size / 2);

        std::vector<size_t> final_hash_values(cloud_in_size);
        std::for_each(std::execution::par_unseq, index0.begin(), index0.end(), [&](const size_t &i)
                      {
        common::V3D loc_xyz_local;
        loc_xyz_local << cur_all_points[i].x, cur_all_points[i].y, cur_all_points[i].z;
        for (int j = 0; j < 3; j++) {
            loc_xyz_local[j] = loc_xyz_local[j] / scan_voxel_size_;
        }
        int64_t ijk0_local = int64_t(floor(loc_xyz_local[0]));
        int64_t ijk1_local = int64_t(floor(loc_xyz_local[1]));
        int64_t ijk2_local = int64_t(floor(loc_xyz_local[2]));
        final_hash_values[i] = ComputeVoxelHash(ijk0_local, ijk1_local, ijk2_local); });

        for (size_t cp = 0; cp < cloud_in_size; cp++)
        {
            auto &voxel = leaves_[final_hash_values[cp]];
            if (voxel.empty())
            {
                voxel.reserve(4);
            }
            voxel.push_back(cur_all_points[cp]);
        }

        // --- 体素内取距离体素中心最近的点 ---
        cloud_down_reg.reserve(leaves_.size());
        for (const auto &leaf : leaves_)
        {
            const auto &voxel_pts = leaf.second;
            if (voxel_pts.empty()) continue;

            // 从体素内第一个点反算体素中心坐标: center = (ijk + 0.5) * scan_voxel_size_
            common::V3D loc_xyz;
            loc_xyz << voxel_pts[0].x, voxel_pts[0].y, voxel_pts[0].z;
            int64_t ijk0 = int64_t(floor(loc_xyz[0] / scan_voxel_size_));
            int64_t ijk1 = int64_t(floor(loc_xyz[1] / scan_voxel_size_));
            int64_t ijk2 = int64_t(floor(loc_xyz[2] / scan_voxel_size_));
            Eigen::Vector3d center;
            center << (ijk0 + 0.5) * scan_voxel_size_,
                      (ijk1 + 0.5) * scan_voxel_size_,
                      (ijk2 + 0.5) * scan_voxel_size_;

            // 寻找距离体素中心最近的点
            double best_dist2 = std::numeric_limits<double>::max();
            size_t best_idx = 0;
            for (size_t i = 0; i < voxel_pts.size(); i++)
            {
                Eigen::Vector3d pt(voxel_pts[i].x, voxel_pts[i].y, voxel_pts[i].z);
                double d2 = (pt - center).squaredNorm();
                if (d2 < best_dist2)
                {
                    best_dist2 = d2;
                    best_idx = i;
                }
            }
            cloud_down_reg.push_back(voxel_pts[best_idx]);
        }

        return;
    }

    bool LaserMapping::time_list(const PointType2 &x, const PointType2 &y) { return (x.curvature < y.curvature); };

    void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                if (msg->header.stamp.toSec() < last_timestamp_lidar_)
                {
                    LOG(ERROR) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.push_back(ptr);
                time_buffer_.push_back(msg->header.stamp.toSec());
                last_timestamp_lidar_ = msg->header.stamp.toSec();
            },
            "Preprocess (Standard)");
        mtx_buffer_.unlock();
    }

    void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {
        mtx_buffer_.lock();
        Timer::Evaluate(
            [&, this]()
            {
                scan_count_++;
                if (msg->header.stamp.toSec() < last_timestamp_lidar_)
                {
                    LOG(WARNING) << "lidar loop back, clear buffer";
                    lidar_buffer_.clear();
                }

                last_timestamp_lidar_ = msg->header.stamp.toSec();

                PointCloudType::Ptr ptr(new PointCloudType());
                preprocess_->Process(msg, ptr);
                lidar_buffer_.emplace_back(ptr);
                time_buffer_.emplace_back(last_timestamp_lidar_);
            },
            "Preprocess (Livox)");

        mtx_buffer_.unlock();
    }

    void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in)
    {
        publish_count_++;
        sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

        msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu_);

        double timestamp = msg->header.stamp.toSec();

        mtx_buffer_.lock();
        if (timestamp < last_timestamp_imu_)
        {
            LOG(WARNING) << "imu loop back, clear buffer";
            imu_buffer_.clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(msg);
        mtx_buffer_.unlock();
    }

    bool LaserMapping::SyncPackages()
    {
        if (lidar_buffer_.empty() || (imu_buffer_.empty()))
        {
            return false;
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed_)
        {
            measures_.lidar_ = lidar_buffer_.front();
            measures_.lidar_bag_time_ = time_buffer_.front();

            double max_point_offset = 0.0;
            for (const auto &point : measures_.lidar_->points)
            {
                max_point_offset = std::max(max_point_offset, double(point.curvature) / 1000.0);
            }

            if (measures_.lidar_->points.size() <= 1)
            {
                LOG(WARNING) << "Too few input point cloud!";
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else if (max_point_offset < 0.5 * lidar_mean_scantime_)
            {
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            }
            else
            {
                scan_num_++;
                lidar_end_time_ = measures_.lidar_bag_time_ + max_point_offset;
                lidar_mean_scantime_ += (max_point_offset - lidar_mean_scantime_) / scan_num_;
            }

            measures_.lidar_end_time_ = lidar_end_time_;
            lidar_pushed_ = true;
        }

        if (last_timestamp_imu_ < lidar_end_time_)
        {
            return false;
        }

        /*** push imu_ data, and pop from imu_ buffer ***/

        measures_.imu_.clear();
        measures_.imu_after_end_.reset();
        while (!imu_buffer_.empty())
        {
            if (imu_buffer_.front()->header.stamp.toSec() > lidar_end_time_)
            {
                // Borrow the right endpoint for interpolation. The next frame
                // still needs this original measurement in the buffer.
                measures_.imu_after_end_ = imu_buffer_.front();
                break;
            }
            measures_.imu_.push_back(imu_buffer_.front());
            imu_buffer_.pop_front();
        }

        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        lidar_pushed_ = false;
        return true;
    }

    void LaserMapping::MapIncremental()
    {
        PointVector points_to_add;

        state_point_ = kf_.get_x();
        const common::M3D map_R_w = state_point_.rot.toRotationMatrix();
        const common::M3D map_R_wl = map_R_w * state_point_.offset_R_L_I.toRotationMatrix();
        const common::V3D map_t_wl = state_point_.pos + map_R_w * state_point_.offset_T_L_I;
        int cur_pts = scan_down_reg_.size();
        points_to_add.clear();
        points_to_add.resize(cur_pts);
        std::vector<size_t> index(cur_pts);
        for (size_t i = 0; i < cur_pts; i++)
        {
            index[i] = i;
        }
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                      {
                          PointType &point_body = scan_down_reg_.at(i);
                          PointType &point_world = points_to_add.at(i);
                          PointBodyToWorld(&point_body, &point_world);

                          // 协方差传播
                          if (map_cov_prop_enabled_ && cov_prop_enabled_)
                          {
                              // 无ZUPT检测时，初始化完成后所有帧均传播协方差
                              bool is_static = false;

                              // 计算雅可比矩阵
                              Eigen::Vector3d p_lidar(point_body.x, point_body.y, point_body.z);

                              // 旋转误差雅可比使用相对于 IMU 原点的点坐标。
                              // 不能加入 state_point_.pos，否则旋转协方差会随
                              // 机器人到世界原点的距离被错误放大。
                              Eigen::Vector3d p_rel_world =
                                  map_R_wl * p_lidar +
                                  map_R_w * state_point_.offset_T_L_I;

                              Eigen::Matrix<double, 3, 6> J =
                                  CovariancePropagation::ComputeJacobian(
                                      p_lidar, map_R_wl, p_rel_world, map_R_w);

                              // 传播状态协方差
                              Eigen::Matrix3d state_prop_cov =
                                  CovariancePropagation::PropagateStateCovariance(J, state_cov_);

                              // 只对位置不确定度进行状态协方差传播
                              point_world.cov_pos = CovariancePropagation::CombineCovariances(
                                  point_world.cov_pos, // LIDAR观测位置不确定度
                                  state_prop_cov,      // 状态传播协方差
                                  is_static);

                              // 几何协方差不参与传播（已在PointBodyToWorld中旋转）
                          } });

        if (voxel_input_pub_en_)
        {
            pcl::PointCloud<pcl::PointXYZI> cloud;
            cloud.reserve(points_to_add.size());
            for (const auto &p : points_to_add)
            {
                pcl::PointXYZI q;
                q.x = p.x; q.y = p.y; q.z = p.z; q.intensity = p.intensity;
                cloud.push_back(q);
            }
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(cloud, msg);
            msg.header.stamp = ros::Time().fromSec(lidar_end_time_);
            msg.header.frame_id = "camera_init";
            pub_voxel_input_.publish(msg);
        }
        ivox_->AddPoints(points_to_add, cross_voxel_merge_en_);
        ivox_->ErasePoints(state_point_.pos, lidar_end_time_ - first_lidar_time_);

        // 协方差统计（调试用）
        // if (map_cov_prop_enabled_ && debug_log_) {
        //     double total_cov_trace = 0;
        //     int valid_count = 0;
        //     for (const auto& pt : points_to_add) {
        //         total_cov_trace += pt.cov().trace();
        //         valid_count++;
        //     }
        //     double mean_cov = valid_count > 0 ? total_cov_trace / valid_count : 0;
        //     std::stringstream ss;
        //     ss << "MapCov: Frame " << frame_num_
        //        << " | Mean cov trace: " << std::scientific << mean_cov;
        //     LOG(INFO) << ss.str();
        // }
    }

    /**
     * Lidar point cloud registration
     * will be called by the eskf custom observation model
     * compute point-to-plane residual here
     * @param s kf state
     * @param ekfom_data H matrix
     */
    void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
    {
        int cnt_pts = scan_down_reg_.size();
        std::vector<size_t> index(cnt_pts);
        for (size_t i = 0; i < cnt_pts; i++)
        {
            index[i] = i;
        }
        state_point_ = kf_.get_x();
        state_cov_ = kf_.get_P().block<6, 6>(0, 0);
        Timer::Evaluate(
            [&, this]()
            {
                const common::M3D R_w = s.rot.toRotationMatrix();
                const common::M3D R_wl = R_w * s.offset_R_L_I.toRotationMatrix();

                {
                    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i)
                                  {
                    PointType &point_body = scan_down_reg_.at(i);
                    PointType &point_world = scan_down_world_.at(i);

                    /* Transform point and its covariances with the same IEKF
                     * iterate state s.  Calling PointBodyToWorld() here would
                     * use the stale member state_point_ for covariance rotation
                     * while the point coordinates use s. */
                    const common::V3D p_body(point_body.x, point_body.y, point_body.z);
                    const common::M3D R_li = s.offset_R_L_I.toRotationMatrix();
                    const common::M3D R_total = R_w * R_li;
                    const common::V3D p_this = R_li * p_body + s.offset_T_L_I;
                    point_world = point_body;
                    const common::V3D p_world = R_w * p_this + s.pos;
                    point_world.x = p_world.x(); point_world.y = p_world.y(); point_world.z = p_world.z();
                    point_world.cov_pos = R_total * point_body.cov_pos * R_total.transpose();
                    point_world.cov_geo = R_total * point_body.cov_geo * R_total.transpose();

                    // 与地图点写入路径保持一致：两个开关均开启时才传播。
                    if (map_cov_prop_enabled_ && cov_prop_enabled_)
                    {
                        Eigen::Vector3d p_lidar_cur(point_body.x, point_body.y, point_body.z);
                        Eigen::Vector3d p_tr = R_wl * p_lidar_cur + R_w * s.offset_T_L_I;
                        Eigen::Matrix<double, 3, 6> J_cur =
                            CovariancePropagation::ComputeJacobian(p_lidar_cur, R_wl, p_tr, R_w);
                        point_world.cov_pos = point_world.cov_pos +
                            CovariancePropagation::PropagateStateCovariance(J_cur, state_cov_);
                    }
                    //                                        if(ekfom_data.converge)
                    {
                        auto &points_near = nearest_points_[i];
                        /** Find the closest surfaces in the map **/
                        ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS, 3);

                        auto &p_sum = plane_coef_[i];
                        if (!points_near.empty()) {
                            PointType p1;
                            common::V3D incident_normal;
                            common::M3D project_2D_cov;
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_2D;

                            p_sum = points_near.at(0);
                            bool gaussian_capacity_reached =
                                p_sum.pt_num >= options::MAX_FEA_NUM;

                            // Track distinct support voxels at the current scan sampling scale.
                            const size_t min_support_voxels = static_cast<size_t>(min_support_voxels_);
                            const size_t max_support_voxels = static_cast<size_t>(max_support_voxels_);
                            // 与 IVox::Pos2Grid() 保持一致：开启时统计实际地图体素，
                            // 关闭时仍按当前帧采样尺度统计支撑体素。
                            const double support_voxel_resolution =
                                support_voxel_use_map_resolution_
                                    ? ivox_options_.resolution_
                                    : (scan_voxel_size_ > 0.0 ? scan_voxel_size_
                                                              : ivox_options_.resolution_);
                            const double inv_voxel_res = 1.0 / support_voxel_resolution;
                            std::vector<size_t> voxel_hashes;
                            voxel_hashes.reserve(max_support_voxels);
                            size_t voxel_count = 0;
                            auto get_voxel_hash = [&](const PointType &pt) -> size_t {
                                int64_t vx = (int64_t)std::floor(pt.x * inv_voxel_res);
                                int64_t vy = (int64_t)std::floor(pt.y * inv_voxel_res);
                                int64_t vz = (int64_t)std::floor(pt.z * inv_voxel_res);
                                return ComputeVoxelHash(vx, vy, vz);
                            };
                            auto add_voxel = [&](size_t hash) {
                                for (size_t k = 0; k < voxel_count; ++k)
                                    if (voxel_hashes[k] == hash) return;
                                if (voxel_count < max_support_voxels) {
                                    voxel_hashes.push_back(hash);
                                    ++voxel_count;
                                }
                            };
                            add_voxel(get_voxel_hash(points_near.at(0)));

                            for (int j = 1; j < points_near.size() && !gaussian_capacity_reached; j++) {
                                // Keep the fitted plane local by limiting its spatial support.
                                if (voxel_count >= max_support_voxels) {
                                    break;
                                }
                                if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                                    add_voxel(get_voxel_hash(points_near.at(j)));
                                    p1 = Merge2(points_near.at(j), p_sum);
                                    p_sum = p1;
                                    gaussian_capacity_reached =
                                        p_sum.pt_num >= options::MAX_FEA_NUM;
                                    continue;
                                } else if (voxel_count < min_support_voxels) {
                                    // 支撑体素不足时继续融合，无需提前计算 p2q。
                                    add_voxel(get_voxel_hash(points_near.at(j)));
                                    p1 = Merge2(points_near.at(j), p_sum);
                                    p_sum = p1;
                                    gaussian_capacity_reached =
                                        p_sum.pt_num >= options::MAX_FEA_NUM;
                                    continue;
                                } else {
                                    // 达到最小支撑体素数后，由 p2q 决定是否继续融合。
                                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov_geo);
                                    Eigen::Vector3d normal_x = es1.eigenvectors().col(0);
                                    Eigen::Vector3d normal_y = es1.eigenvectors().col(1);
                                    Eigen::Vector3d normal_z = es1.eigenvectors().col(2);
                                    Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                         Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                    double p2q =
                                        pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                        pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));

                                    bool cond_p2q = p2q > t_stop_pseudo_merge_ * t_stop_pseudo_merge_;
                                    if (cond_p2q) {
                                        add_voxel(get_voxel_hash(points_near.at(j)));
                                        p1 = Merge2(points_near.at(j), p_sum);
                                        p_sum = p1;
                                        continue;
                                    } else {
                                        for (int k = j; k < points_near.size();) points_near.pop_back();
                                    }
                                }
                            }
                            // // 判断匹配类型
                            // if (p_sum.pt_num < options::MIN_NUM_MATCH_POINTS) {
                            //     // 点数不足 → 丢弃
                            //     points_near.clear();
                            // } else {
                                // 点数足够，使用完整判断
                                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es1(p_sum.cov_geo);
                                Eigen::Vector3d normal_x = es1.eigenvectors().col(0);
                                Eigen::Vector3d normal_y = es1.eigenvectors().col(1);
                                Eigen::Vector3d normal_z = es1.eigenvectors().col(2);
                                Eigen::Vector3d pq = Eigen::Vector3d(p_sum.x, p_sum.y, p_sum.z) -
                                                     Eigen::Vector3d(point_world.x, point_world.y, point_world.z);
                                double p2q = pow(normal_y.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(1)) +
                                             pow(normal_z.transpose() * pq, 2) * (1.0f / es1.eigenvalues().real()(2));


                                if (!(
                                    // (es1.eigenvalues().real()(1) > 0.025)
                                // (es1.eigenvalues().real()(1) * es1.eigenvalues().real()(2) / (es1.eigenvalues().real()(1) + es1.eigenvalues().real()(2)) > 0.025)
                                // 容量上限只限制后续融合；平面有效性仍要求达到
                                // 当前帧采样尺度下的最小空间支撑。
                                (voxel_count >= min_support_voxels)
                                // (es1.eigenvalues().real()(1) * es1.eigenvalues().real()(2) / (es1.eigenvalues().real()(1) + es1.eigenvalues().real()(2)) > 0.03125)
                                && (p2q <= t_stop_pseudo_merge_ * t_stop_pseudo_merge_)))
                                // && (p2q <= 1)
                                // && (p2q <= 2.0)
                                // && (es1.eigenvalues().real()(0) < 10.0f/4.0f * 0.01)
                                // // ||((es1.eigenvalues().real()(0))/(es1.eigenvalues().real()(0)+es1.eigenvalues().real()(1)+es1.eigenvalues().real()(2)) <= tau_planar_)
                            // if ((d < options::T_MAL)
                                {
                                    points_near.clear();
                                }
                            // }
                        }
                    } });
                }
            },
            "    ObsModel (Lidar Match)");

        effect_feat_num_ = 0;

        corr_pts_.resize(cnt_pts);
        corr_norm_.resize(cnt_pts);
        for (int i = 0; i < cnt_pts; i++)
        {
            if (!nearest_points_[i].empty())
            {
                point_selected_surf_[i] = true;
            }
            else
                point_selected_surf_[i] = false;

            if (point_selected_surf_[i])
            {
                point_selected_idx_[effect_feat_num_] = i;
                corr_norm_[effect_feat_num_] = plane_coef_[i];
                corr_pts_[effect_feat_num_] = scan_down_reg_.at(i);
                effect_feat_num_++;
            }
        }
        corr_pts_.resize(effect_feat_num_);
        corr_norm_.resize(effect_feat_num_);
        if (obs_error_viz_en_) obs_errors_.resize(effect_feat_num_);

        // 预计算 corr_norm_ cov_geo 的特征分解，IEKF 迭代中直接复用
        corr_norm_eig_cache_.resize(effect_feat_num_);
        for (size_t i = 0; i < effect_feat_num_; i++)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(corr_norm_[i].cov_geo);
            corr_norm_eig_cache_[i].eigenvalues = es.eigenvalues();
            corr_norm_eig_cache_[i].eigenvectors = es.eigenvectors();
        }

        if (effect_feat_num_ < 1)
        {
            ekfom_data.valid = false;
            ROS_WARN("No Effective Points!");
            return;
        }
        Timer::Evaluate(
            [&, this]()
            {
                ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12); // 23
                ekfom_data.h.resize(effect_feat_num_);
                ekfom_data.R_inv.resize(effect_feat_num_, 1);
                std::vector<size_t> index2(effect_feat_num_);
                for (size_t i = 0; i < effect_feat_num_; i++)
                {
                    index2[i] = i;
                }
                const common::M3D off_R = s.offset_R_L_I.toRotationMatrix();
                const common::V3D off_t = s.offset_T_L_I;
                const common::M3D Rt = s.rot.toRotationMatrix().transpose();

                std::for_each(std::execution::par_unseq, index2.begin(), index2.end(), [&](const size_t &i)
                              {
                                common::V3D point_this_be(corr_pts_[i].x, corr_pts_[i].y, corr_pts_[i].z);
                                common::M3D point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                                common::V3D point_this = off_R * point_this_be + off_t;
                                common::M3D point_crossmat = SKEW_SYM_MATRIX(point_this);
                                common::V3D point_world = s.rot.toRotationMatrix() * point_this + s.pos;

                                // 从缓存读取特征分解（无需重复 SelfAdjointEigenSolver）
                                const auto &eig_cache = corr_norm_eig_cache_[i];
                                common::V3D norm_p = eig_cache.eigenvectors.col(0);  // v0: 法向量
                                common::V3D v1 = eig_cache.eigenvectors.col(1);      // v1: 第二特征向量
                                common::V3D v2 = eig_cache.eigenvectors.col(2);      // v2: 第三特征向量

                                double lambda0 = eig_cache.eigenvalues(0);  // 最小特征值
                                double lambda1 = eig_cache.eigenvalues(1);
                                double lambda2 = eig_cache.eigenvalues(2);

                                common::V3D map_q(corr_norm_[i].x, corr_norm_[i].y, corr_norm_[i].z);

                                // ========== 点到面ICP ==========
                                ekfom_data.h_x.block<1, 3>(i, 0) << norm_p.transpose();
                                ekfom_data.h_x.block<1, 3>(i, 3) << -norm_p.transpose() * (s.rot.toRotationMatrix() * point_crossmat);
                                ekfom_data.h(i) = norm_p.transpose() * (map_q - point_world);

                                // 1. 当前帧点测量噪声（体坐标系观测误差变换到世界系）
                                double thickness_p = (norm_p.transpose() * s.rot.toRotationMatrix() * off_R ) * corr_pts_[i].cov_pos * (off_R.transpose() * s.rot.toRotationMatrix().transpose() * norm_p);
                                // 2. 当前点云虚拟中心位置不确定度
                                double thickness_pos = (norm_p.transpose() * (corr_norm_[i].cov_pos) * norm_p);

                                // 3. 法向量不确定度
                                common::V3D v0 = norm_p;
                                double v0_Sigpos_v0 = v0.transpose() * (corr_norm_[i].cov_pos) * v0;
                                double v1_Sigpos_v1 = v1.transpose() * (corr_norm_[i].cov_pos) * v1;
                                double v2_Sigpos_v2 = v2.transpose() * (corr_norm_[i].cov_pos) * v2;
                                double v1_Sigpos_v2 = v1.transpose() * (corr_norm_[i].cov_pos) * v2;

                                common::V3D delta_p = map_q - point_world;
                                double proj_v1 = delta_p.transpose() * v1;
                                double proj_v2 = delta_p.transpose() * v2;
                                double N = corr_norm_[i].pt_num;
                                double N_ratio = N / (N - 1.0);
                                // double N_ratio = 1.0;

                                // // 公式1 (无偏, 无交叉项): term = (...) / (Δλ)²
                                // double term1 = (lambda1 * v0_Sigpos_v0 + lambda0 * v1_Sigpos_v1) / ((lambda1 - lambda0) * (lambda1 - lambda0));
                                // double term2 = (lambda2 * v0_Sigpos_v0 + lambda0 * v2_Sigpos_v2) / ((lambda2 - lambda0) * (lambda2 - lambda0));
                                // double thickness_norm = term1 * proj_v1 * proj_v1 + term2 * proj_v2 * proj_v2;

                                // 公式2 (有偏修正 N/(N-1) + 交叉项)
                                // 防护：特征值间隙过小时法向量不确定度公式失效，需 clamp 分母
                                constexpr double kMinEigenGap = 1e-8;
                                double gap1 = lambda1 - lambda0;
                                double gap2 = lambda2 - lambda0;
                                gap1 = gap1 < kMinEigenGap ? kMinEigenGap : gap1;
                                gap2 = gap2 < kMinEigenGap ? kMinEigenGap : gap2;

                                double term1 = N_ratio * (lambda1 * v0_Sigpos_v0 + lambda0 * v1_Sigpos_v1) / (gap1 * gap1);
                                double term2 = N_ratio * (lambda2 * v0_Sigpos_v0 + lambda0 * v2_Sigpos_v2) / (gap2 * gap2);
                                double term_cross = N_ratio * lambda0 * v1_Sigpos_v2 / (gap2 * gap1);
                                double thickness_norm = term1 * proj_v1 * proj_v1 + term2 * proj_v2 * proj_v2
                                                      + 2.0 * term_cross * proj_v1 * proj_v2;

                                double thickness = thickness_p + thickness_pos + thickness_norm;
                                // double thickness1 = exp(norm_p.transpose() * (map_q - point_world) * norm_p.transpose() * (map_q - point_world))*lambda0;
                                // double thickness = exp(0.01) * (0.001 + lambda0);
                                // double thickness = 0.001;
                                // double thickness = thickness_p + thickness_pos;
                                // // debug: 每100帧打印前5个点的三分量
                                // if (obs_error_viz_en_ && i < 5 && frame_num_ % 100 == 0) {
                                //     std::ostringstream oss;
                                //     oss << "[ObsErr] frame=" << frame_num_ << " pt=" << i
                                //         << " tp=" << std::scientific << std::setprecision(6) << thickness_p
                                //         << " tpos=" << thickness_pos
                                //         << " tnorm=" << thickness_norm
                                //         << " total=" << thickness;
                                //     LOG(INFO) << oss.str();
                                // }

                                double residual = norm_p.transpose() * (map_q - point_world);
                                double residual_variance = thickness;
                                residual_variance = std::max(residual_variance, residual_variance_floor_);
                                double gate = residual_gate_sigma_ * std::sqrt(residual_variance);
                                if (!std::isfinite(residual_variance) || !std::isfinite(residual) || std::abs(residual) > gate)
                                {
                                    ekfom_data.h_x.row(i).setZero();
                                    ekfom_data.h(i) = 0.0;
                                    ekfom_data.R_inv(i) = 0.0;
                                }
                                else
                                {
                                ekfom_data.R_inv(i) = 1.0 / thickness;

                                if (obs_error_viz_en_) {
                                    obs_errors_[i](0) = thickness;
                                    obs_errors_[i](1) = 0.001;
                                    obs_errors_[i](2) = exp(0.01) * (0.001 + lambda0);
                                }
                                }
                            });
            },
            "    ObsModel (IEKF Build Jacobian)");
    }

    /////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

    void LaserMapping::WriteConsistencyLog()
    {
        if (!consistency_log_en_ || !fout_consistency_)
            return;

        const auto &diag = kf_.get_last_update_diagnostics();
        const auto pose_cov = kf_.get_P().block<6, 6>(0, 0);
        const auto &q = state_point_.rot;
        const double nan = std::numeric_limits<double>::quiet_NaN();

        fout_consistency_ << std::fixed << std::setprecision(9)
                          << lidar_end_time_ << ","
                          << state_point_.pos.x() << "," << state_point_.pos.y() << ","
                          << state_point_.pos.z() << "," << q.x() << "," << q.y() << ","
                          << q.z() << "," << q.w();
        fout_consistency_ << std::scientific << std::setprecision(12);
        for (int row = 0; row < 6; ++row)
            for (int col = 0; col < 6; ++col)
                fout_consistency_ << "," << pose_cov(row, col);
        fout_consistency_ << "," << (diag.valid ? diag.nis : nan)
                          << "," << (diag.valid ? diag.innovation_dof : 0)
                          << "," << (diag.valid ? diag.nis_per_dof : nan)
                          << "," << (diag.valid ? diag.whitened_residual_sum : nan)
                          << "\n";
    }

    void LaserMapping::PublishPath(const ros::Publisher pub_path)
    {
        SetPosestamp(msg_body_pose_);
        msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
        msg_body_pose_.header.frame_id = "camera_init";

        /*** if path is too large, the rviz will crash ***/
        path_.poses.push_back(msg_body_pose_);
        if (run_in_offline_ == false)
        {
            pub_path.publish(path_);
        }
    }

    void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped)
    {
        odom_aft_mapped_.header.frame_id = "camera_init";
        odom_aft_mapped_.child_frame_id = "body";
        odom_aft_mapped_.header.stamp = ros::Time().fromSec(lidar_end_time_); // ros::Time().fromSec(lidar_end_time_);
        SetPosestamp(odom_aft_mapped_.pose);

        auto P = kf_.get_P();
        for (int i = 0; i < 6; i++)
        {
            odom_aft_mapped_.pose.covariance[i * 6 + 0] = P(i, 0);
            odom_aft_mapped_.pose.covariance[i * 6 + 1] = P(i, 1);
            odom_aft_mapped_.pose.covariance[i * 6 + 2] = P(i, 2);
            odom_aft_mapped_.pose.covariance[i * 6 + 3] = P(i, 3);
            odom_aft_mapped_.pose.covariance[i * 6 + 4] = P(i, 4);
            odom_aft_mapped_.pose.covariance[i * 6 + 5] = P(i, 5);
        }
        pub_odom_aft_mapped.publish(odom_aft_mapped_);
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odom_aft_mapped_.pose.pose.position.x, odom_aft_mapped_.pose.pose.position.y,
                                        odom_aft_mapped_.pose.pose.position.z));
        q.setW(odom_aft_mapped_.pose.pose.orientation.w);
        q.setX(odom_aft_mapped_.pose.pose.orientation.x);
        q.setY(odom_aft_mapped_.pose.pose.orientation.y);
        q.setZ(odom_aft_mapped_.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, "camera_init", "body"));
    }

    void LaserMapping::PublishFrameWorld()
    {
        PointCloudType::Ptr laserCloudWorld;

        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld.reset(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++)
        {
            common::V3D p_body(laserCloudFullRes->points.at(i).x, laserCloudFullRes->points.at(i).y,
                               laserCloudFullRes->points.at(i).z);
            common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                                 state_point_.pos);

            laserCloudWorld->points[i].x = p_global(0);
            laserCloudWorld->points[i].y = p_global(1);
            laserCloudWorld->points[i].z = p_global(2);
            laserCloudWorld->points[i].intensity = laserCloudFullRes->points.at(i).intensity;
            laserCloudWorld->points[i].curvature = laserCloudFullRes->points.at(i).curvature;
            laserCloudWorld->points[i].normal_x = laserCloudFullRes->points.at(i).normal_x;
            laserCloudWorld->points[i].normal_y = laserCloudFullRes->points.at(i).normal_y;
            laserCloudWorld->points[i].normal_z = laserCloudFullRes->points.at(i).normal_z;
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_world_.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    void LaserMapping::GetRainbowColor(float value, common::V3F &color)
    {
        // this is HSV color palette with hue values going only from 0.0 to 0.833333.

        value = std::min(value, 1.0f);
        value = std::max(value, 0.0f);

        float h = value * 5.0f + 1.0f;
        int i = floor(h);
        float f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        float n = 1 - f;

        if (i <= 1)
            color[0] = n, color[1] = 0, color[2] = 1;
        else if (i == 2)
            color[0] = 0, color[1] = n, color[2] = 1;
        else if (i == 3)
            color[0] = 0, color[1] = 1, color[2] = n;
        else if (i == 4)
            color[0] = n, color[1] = 1, color[2] = 0;
        else if (i >= 5)
            color[0] = 1, color[1] = n, color[2] = 0;
    }

    void LaserMapping::PublishMap()
    {
        ivox_->GetMapPoints(map_points_);
        int size = map_points_.size();
        LOG(INFO) << "map size " << size << " voxel size " << ivox_->NumValidGrids() << " average point each voxel "
                  << float(size) / float(ivox_->NumValidGrids());
        if (size == 0)
            return;
        sensor_msgs::PointCloud2 laserCloudMap;
        PointCloudType::Ptr map_pub(new PointCloudType());
        PointType2 body_p;
        map_pub->clear();
        double max_intensity = 0, min_z = 10000.0, max_z = -10000.0;
        min_z = state_point_.pos.z() + 3;
        max_z = state_point_.pos.z() + 30;
        {
            common::M3D world_cov;
            visualization_msgs::Marker p_cov;
            p_cov.type = visualization_msgs::Marker::SPHERE;
            p_cov.action = visualization_msgs::Marker::ADD;
            p_cov.header.frame_id = "camera_init";
            p_cov.header.stamp = ros::Time().fromSec(lidar_end_time_);
            p_cov.lifetime = ros::Duration(0); // 3 x process time
            pa_cov.markers.clear();
            common::V3D state_pos(state_point_.pos);
            common::V3D map_p;
            double dis = 0;
            int pt_cnt = -1;
            for (auto it = map_points_.begin(); it != map_points_.end(); it++)
            {
                world_cov = it->cov();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();

                body_p.x = it->x;
                body_p.y = it->y;
                body_p.z = it->z;
                body_p.curvature = it->pt_num;
                body_p.intensity = it->intensity;
                // 使用非线性变换增强协方差对比度
                // sqrt(ev) 的平方 = ev，放大了协方差差异
                double std_dev = sqrt(fabs(ev_cov3(2)));
                body_p.normal_z = std_dev * std_dev * 5.0; // 平方放大并乘以系数5
                map_pub->push_back(body_p);
                if (gaussian_publish_en_)
                {
                    if (it->intensity > max_intensity)
                        max_intensity = it->intensity;

                    if (it->pt_num <= gaussian_pub_min_cnt_)
                        continue;
                    map_p << it->x, it->y, it->z;
                    dis = (map_p.x() - state_pos.x()) * (map_p.x() - state_pos.x()) +
                          (map_p.y() - state_pos.y()) * (map_p.y() - state_pos.y()) +
                          (map_p.z() - state_pos.z()) * (map_p.z() - state_pos.z());

                    if (dis > gaussian_pub_dis_ * gaussian_pub_dis_)
                        continue; // pub local gmm only

                    p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter 2 sigema
                    p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                    p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                    p_cov.pose.position.x = it->x;
                    p_cov.pose.position.y = it->y;
                    p_cov.pose.position.z = it->z;
                    p_cov.id = pa_cov.markers.size();
                    p_cov.color.r = 0.53f;
                    p_cov.color.g = 0.81f;
                    p_cov.color.b = 1.0f;
                    p_cov.ns = "gaussian_map";

                    p_cov.color.a = 0.70f;
                    Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                    Eigen::Quaterniond eq3(rotation3);
                    p_cov.pose.orientation.w = eq3.w();
                    p_cov.pose.orientation.x = eq3.x();
                    p_cov.pose.orientation.y = eq3.y();
                    p_cov.pose.orientation.z = eq3.z();
                    pa_cov.markers.push_back(p_cov);
                }
            }
            map_cov_pub.publish(pa_cov);
        }

        pcl::toROSMsg(*map_pub, laserCloudMap);
        laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudMap.header.frame_id = "camera_init";
        pub_map.publish(laserCloudMap);
    }

    void LaserMapping::PublishMatchedGaussian()
    {
        visualization_msgs::MarkerArray ma;

        if (obs_error_viz_en_)
        {
            // ========== 观测误差模型可视化：同方向并排三线对比 ==========
            const char* ns_names[3] = {"error_model_1", "error_model_2", "error_model_3"};
            for (int k = 0; k < 3; k++)
            {
                visualization_msgs::Marker del;
                del.action = visualization_msgs::Marker::DELETEALL;
                del.ns = ns_names[k];
                del.header.frame_id = "camera_init";
                del.header.stamp = ros::Time().fromSec(lidar_end_time_);
                ma.markers.push_back(del);
            }

            // 三根线沿 z 方向，y 方向微小偏移并排，长度 = sqrt(thickness)
            struct ModelConfig { const char* ns; float r, g, b; int idx; double y_off; };
            ModelConfig models[3] = {
                {"error_model_1", 1.0f, 0.0f, 0.0f, 0, -0.02},  // 红: full model
                {"error_model_2", 0.0f, 1.0f, 0.0f, 1,  0.00},  // 绿: constant 0.001
                {"error_model_3", 0.0f, 0.0f, 1.0f, 2,  0.02},  // 蓝: exp(0.01)*(0.001+lambda0)
            };

            for (int k = 0; k < 3; k++)
            {
                visualization_msgs::Marker line;
                line.type = visualization_msgs::Marker::LINE_LIST;
                line.action = visualization_msgs::Marker::ADD;
                line.header.frame_id = "camera_init";
                line.header.stamp = ros::Time().fromSec(lidar_end_time_);
                line.lifetime = ros::Duration();
                line.ns = models[k].ns;
                line.id = 0;
                line.color.r = models[k].r;
                line.color.g = models[k].g;
                line.color.b = models[k].b;
                line.color.a = 0.9f;
                line.scale.x = 0.02;
                line.pose.orientation.w = 1.0;

                line.points.reserve(effect_feat_num_ * 2);

                for (int i = 0; i < effect_feat_num_; i++)
                {
                    geometry_msgs::Point p0, p1;
                    p0.x = corr_norm_[i].x;
                    p0.y = corr_norm_[i].y + models[k].y_off;
                    p0.z = corr_norm_[i].z;

                    double len = std::sqrt(std::max(obs_errors_[i](models[k].idx), 0.0));
                    p1.x = p0.x;
                    p1.y = p0.y;
                    p1.z = p0.z + len;

                    line.points.push_back(p0);
                    line.points.push_back(p1);
                }

                ma.markers.push_back(line);
            }

            matched_gaussian_pub.publish(ma);
            return;
        }

        // ========== 原始高斯基元可视化 ==========
        // 先发送清除 marker
        visualization_msgs::Marker del;
        del.action = visualization_msgs::Marker::DELETEALL;
        del.ns = "matched_gaussian";
        del.header.frame_id = "camera_init";
        del.header.stamp = ros::Time().fromSec(lidar_end_time_);
        ma.markers.push_back(del);

        visualization_msgs::Marker m;
        m.type = visualization_msgs::Marker::SPHERE;
        m.action = visualization_msgs::Marker::ADD;
        m.header.frame_id = "camera_init";
        m.header.stamp = ros::Time().fromSec(lidar_end_time_);
        m.lifetime = ros::Duration();
        m.ns = "matched_gaussian";
        m.color.r = 1.0f;
        m.color.g = 0.82f;
        m.color.b = 0.55f;
        m.color.a = 0.35f;

        ma.markers.reserve(effect_feat_num_ + 1);

        for (int i = 0; i < effect_feat_num_; i++)
        {
            const auto &eig_cache = corr_norm_eig_cache_[i];
            Eigen::Matrix3d evecs = eig_cache.eigenvectors.transpose();
            evecs.row(2) = evecs.row(0).cross(evecs.row(1));
            common::V3D ev = eig_cache.eigenvalues;

            m.pose.position.x = corr_norm_[i].x;
            m.pose.position.y = corr_norm_[i].y;
            m.pose.position.z = corr_norm_[i].z;
            m.id = i + 1;

            // 面片尺寸：1σ 直径
            m.scale.x = 2 * sqrt(fabs(ev(0)));
            m.scale.y = 2 * sqrt(fabs(ev(1)));
            m.scale.z = 2 * sqrt(fabs(ev(2)));

            Eigen::Matrix3d rot = evecs.transpose();
            Eigen::Quaterniond q(rot);
            m.pose.orientation.w = q.w();
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();

            ma.markers.push_back(m);
        }
        matched_gaussian_pub.publish(ma);
    }

    void LaserMapping::PublishVoxelDebug()
    {
        // ====================================================================
        // 高斯基元跨体素验证：逐个检查高斯 3σ椭球是否超过 1 个体素
        //   voxel_grid:  体素参考网格 (浅灰线框)
        //   voxel_cov:   高斯的 cov_geo 3σ椭球
        //                绿 = ≤1个体素 (无问题) / 红 = >1个体素 (跨体素)
        //   voxel_pt:    高斯质心 (白点)
        // ====================================================================
        ivox_->GetMapPoints(map_points_);
        if (map_points_.empty()) return;

        visualization_msgs::MarkerArray ma;

        double resolution = ivox_options_.resolution_;
        double inv_res = 1.0 / resolution;
        int marker_id = 0;
        double viz_range = 20.0;
        common::V3D robot_pos = state_point_.pos;
        double viz_range_sq = viz_range * viz_range;

        auto set_color = [](visualization_msgs::Marker &m, float r, float g, float b, float a) {
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
        };

        int total_pts = 0, pts_with_cov = 0, pts_cross_voxel = 0;
        std::unordered_set<size_t> voxel_drawn;

        for (const auto &pt : map_points_)
        {
            double dx = pt.x - robot_pos.x();
            double dy = pt.y - robot_pos.y();
            double dz = pt.z - robot_pos.z();
            if (dx*dx + dy*dy + dz*dz > viz_range_sq) continue;
            total_pts++;

            // --- 体素参考框 (浅灰，去重) ---
            int64_t vx = (int64_t)std::floor(pt.x * inv_res);
            int64_t vy = (int64_t)std::floor(pt.y * inv_res);
            int64_t vz = (int64_t)std::floor(pt.z * inv_res);
            size_t vhash = ComputeVoxelHash(vx, vy, vz);
            if (voxel_drawn.count(vhash) == 0) {
                voxel_drawn.insert(vhash);
                visualization_msgs::Marker grid;
                grid.header.frame_id = "camera_init";
                grid.header.stamp = ros::Time().fromSec(lidar_end_time_);
                grid.ns = "voxel_grid";
                grid.id = marker_id++;
                grid.type = visualization_msgs::Marker::CUBE;
                grid.action = visualization_msgs::Marker::ADD;
                grid.pose.position.x = (vx + 0.5) * resolution;
                grid.pose.position.y = (vy + 0.5) * resolution;
                grid.pose.position.z = (vz + 0.5) * resolution;
                grid.pose.orientation.w = 1.0;
                grid.scale.x = resolution * 0.95;
                grid.scale.y = resolution * 0.95;
                grid.scale.z = resolution * 0.95;
                grid.lifetime = ros::Duration(0);  // 永久保持
                set_color(grid, 0.35f, 0.35f, 0.35f, 0.06f);
                ma.markers.push_back(grid);
            }

            // --- 高斯质心 (白点) ---
            {
                visualization_msgs::Marker c;
                c.header.frame_id = "camera_init";
                c.header.stamp = ros::Time().fromSec(lidar_end_time_);
                c.ns = "voxel_pt";
                c.id = marker_id++;
                c.type = visualization_msgs::Marker::SPHERE;
                c.action = visualization_msgs::Marker::ADD;
                c.pose.position.x = pt.x;
                c.pose.position.y = pt.y;
                c.pose.position.z = pt.z;
                c.pose.orientation.w = 1.0;
                c.scale.x = 0.04; c.scale.y = 0.04; c.scale.z = 0.04;
                c.lifetime = ros::Duration(0);
                set_color(c, 1.0f, 1.0f, 1.0f, 0.6f);
                ma.markers.push_back(c);
            }

            // --- 跨体素判断 (仅 pt_num>1 有 cov_geo) ---
            if (pt.pt_num < 2) continue;
            pts_with_cov++;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(pt.cov_geo);
            Eigen::Vector3d ev = es.eigenvalues().real();
            Eigen::Matrix3d evecs = es.eigenvectors().real();

            // 最大 3σ 直径 vs 体素大小
            double max_3sigma_diam = 2.0 * 3.0 * std::sqrt(
                std::max({std::fabs(ev(0)), std::fabs(ev(1)), std::fabs(ev(2))}));
            bool cross_voxel = (max_3sigma_diam > resolution);

            // 只画有问题的 (跨体素) 椭球，绿色正常的不画 → 画面清晰
            if (!cross_voxel) continue;
            pts_cross_voxel++;

            visualization_msgs::Marker ellipsoid;
            ellipsoid.header.frame_id = "camera_init";
            ellipsoid.header.stamp = ros::Time().fromSec(lidar_end_time_);
            ellipsoid.ns = "voxel_cov";
            ellipsoid.id = marker_id++;
            ellipsoid.type = visualization_msgs::Marker::SPHERE;
            ellipsoid.action = visualization_msgs::Marker::ADD;
            ellipsoid.pose.position.x = pt.x;
            ellipsoid.pose.position.y = pt.y;
            ellipsoid.pose.position.z = pt.z;
            evecs.col(2) = evecs.col(0).cross(evecs.col(1));
            Eigen::Quaterniond q(evecs.transpose());
            ellipsoid.pose.orientation.w = q.w();
            ellipsoid.pose.orientation.x = q.x();
            ellipsoid.pose.orientation.y = q.y();
            ellipsoid.pose.orientation.z = q.z();
            ellipsoid.scale.x = 2.0 * 3.0 * std::sqrt(std::fabs(ev(0)));
            ellipsoid.scale.y = 2.0 * 3.0 * std::sqrt(std::fabs(ev(1)));
            ellipsoid.scale.z = 2.0 * 3.0 * std::sqrt(std::fabs(ev(2)));
            ellipsoid.lifetime = ros::Duration(0);
            set_color(ellipsoid, 1.0f, 0.15f, 0.0f, 0.55f);  // 红: 仅跨体素高斯
            ma.markers.push_back(ellipsoid);
        }

        voxel_debug_pub.publish(ma);
        LOG(INFO) << "[VoxelDebug] total=" << total_pts << " with_cov=" << pts_with_cov
                  << " cross_voxel=" << pts_cross_voxel
                  << " (" << (pts_with_cov > 0 ? 100.0 * pts_cross_voxel / pts_with_cov : 0)
                  << "%) | voxels=" << voxel_drawn.size();
    }

    void LaserMapping::PublishFrameRegWorld(const ros::Publisher &pub_laser_cloud_reg_world)
    {
        int size = scan_down_reg_.size();
        PointCloudType::Ptr laser_cloud(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++)
        {
            PointBodyToWorldPub(&scan_down_reg_.at(i), &laser_cloud->points[i]);
            if (point_selected_surf_[i])
                laser_cloud->points[i].normal_x = plane_coef_[i].pt_num;
            else
                laser_cloud->points[i].normal_x = 0;
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laser_cloud, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "camera_init";
        pub_laser_cloud_reg_world.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
        if (gaussian_publish_en_)
        {
            common::M3D world_cov;
            visualization_msgs::MarkerArray pa_scan_cov;
            visualization_msgs::Marker p_cov;

            p_cov.type = visualization_msgs::Marker::SPHERE;
            p_cov.action = visualization_msgs::Marker::ADD;
            p_cov.header.frame_id = "camera_init";
            p_cov.header.stamp = ros::Time().fromSec(lidar_end_time_);
            p_cov.lifetime = ros::Duration();
            pa_scan_cov.markers.clear();

            size = scan_down_reg_.size();
            scan_down_reg_.resize(size);
            for (int i = 0; i < scan_down_reg_.size(); i++)
            {
                PointBodyToWorld(&scan_down_reg_[i], &scan_down_world_[i]);
                common::V3D p_body(scan_down_world_[i].x, scan_down_world_[i].y, scan_down_world_[i].z);
                world_cov = scan_down_world_.at(i).cov().block<3, 3>(0, 0);
                p_cov.pose.position.x = scan_down_world_[i].x;
                p_cov.pose.position.y = scan_down_world_[i].y;
                p_cov.pose.position.z = scan_down_world_[i].z;
                p_cov.id = i;
                p_cov.ns = "scan_cov";
                p_cov.color.r = 1;
                p_cov.color.g = 0;
                p_cov.color.b = 0;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es_cov3(world_cov);
                Eigen::Matrix3d sorted_evecs_cov3 = es_cov3.eigenvectors().real().transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = es_cov3.eigenvalues().real();
                p_cov.scale.x = 12 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 12 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 12 * sqrt(fabs(ev_cov3(2)));
                p_cov.color.a = 1;
                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();
                pa_scan_cov.markers.push_back(p_cov);
            }

            size = corr_norm_.size();
            for (int i = 0; i < size; i++)
            {
                p_cov.pose.position.x = corr_norm_.at(i).x;
                p_cov.pose.position.y = corr_norm_.at(i).y;
                p_cov.pose.position.z = corr_norm_.at(i).z;
                p_cov.id = i;

                const auto &eig_cache = corr_norm_eig_cache_[i];
                Eigen::Matrix3d sorted_evecs_cov3 = eig_cache.eigenvectors.transpose();
                sorted_evecs_cov3.row(2) = sorted_evecs_cov3.row(0).cross(sorted_evecs_cov3.row(1));
                common::V3D ev_cov3 = eig_cache.eigenvalues;

                Eigen::Matrix3d rotation3 = sorted_evecs_cov3.transpose();
                Eigen::Quaterniond eq3(rotation3);
                p_cov.pose.orientation.w = eq3.w();
                p_cov.pose.orientation.x = eq3.x();
                p_cov.pose.orientation.y = eq3.y();
                p_cov.pose.orientation.z = eq3.z();

                p_cov.scale.x = 4 * sqrt(fabs(ev_cov3(0))); // scale is diameter
                p_cov.scale.y = 4 * sqrt(fabs(ev_cov3(1)));
                p_cov.scale.z = 4 * sqrt(fabs(ev_cov3(2)));
                p_cov.color.r = 0;
                p_cov.color.g = 0;
                p_cov.color.b = 1;
                p_cov.color.a = 0.3;
                p_cov.ns = "cor_cov";
                p_cov.type = visualization_msgs::Marker::SPHERE;
                pa_scan_cov.markers.push_back(p_cov);

                rotation3.setZero();
                rotation3.row(0) = sorted_evecs_cov3.row(0);
                if (rotation3.row(0).sum() < 0)
                    rotation3 = -rotation3;
                Eigen::Quaterniond eq(rotation3.transpose());
                eq.normalize();
                p_cov.pose.orientation.w = eq.w();
                p_cov.pose.orientation.x = eq.x();
                p_cov.pose.orientation.y = eq.y();
                p_cov.pose.orientation.z = eq.z();

                p_cov.scale.x = 1;
                p_cov.scale.y = 0.2;
                p_cov.scale.z = 0.2;
                p_cov.color.r = 1;
                p_cov.color.g = 1;
                p_cov.color.b = 0;
                p_cov.color.a = 1;
                p_cov.ns = "cor_normal";
                p_cov.type = visualization_msgs::Marker::ARROW;
                pa_scan_cov.markers.push_back(p_cov);
            }
            p_cov.type = visualization_msgs::Marker::LINE_LIST;
            p_cov.ns = "scan_cor_line";
            p_cov.pose.position.x = 0;
            p_cov.pose.position.y = 0;
            p_cov.pose.position.z = 0;
            p_cov.pose.orientation.w = 1;
            p_cov.pose.orientation.x = 0;
            p_cov.pose.orientation.y = 0;
            p_cov.pose.orientation.z = 0;
            p_cov.color.r = 0;
            p_cov.color.g = 1;
            p_cov.color.b = 0;
            p_cov.color.a = 1;
            p_cov.scale.x = 0.03;

            for (int i = 0; i < size; i++)
            {
                geometry_msgs::Point p;
                PointType eff_world;
                p_cov.id = i;
                p.x = corr_norm_[i].x;
                p.y = corr_norm_[i].y;
                p.z = corr_norm_[i].z;
                p_cov.points.push_back(p);
                PointBodyToWorld(&corr_pts_.at(i), &eff_world);
                p.x = eff_world.x;
                p.y = eff_world.y;
                p.z = eff_world.z;
                p_cov.points.push_back(p);
                pa_scan_cov.markers.push_back(p_cov);
                p_cov.points.clear();
            }
            point_cov_pub.publish(pa_scan_cov);
        }
    }

    void LaserMapping::Savetrajectory(const std::string &traj_file)
    {
        std::ofstream ofs;
        ofs.open(traj_file, std::ios::out);
        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open traj_file: " << traj_file;
            return;
        }

        ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
        for (const auto &p : path_.poses)
        {
            ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
                << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " " << p.pose.orientation.x
                << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " " << p.pose.orientation.w << std::endl;
        }

        ofs.close();

        // ofs kitti
        std::string kitti_file = (std::string(std::string(ROOT_DIR) + "Log/" + "kitti.txt"));
        ofs.open(kitti_file, std::ios::out);

        if (!ofs.is_open())
        {
            LOG(ERROR) << "Failed to open kitti_file: " << kitti_file;
            return;
        }

        for (const auto &p : path_.poses)
        {
            SO3 r_matrix;
            r_matrix.x() = p.pose.orientation.x;
            r_matrix.y() = p.pose.orientation.y;
            r_matrix.z() = p.pose.orientation.z;
            r_matrix.w() = p.pose.orientation.w;
            Eigen::MatrixXd kitti_pose;
            kitti_pose.resize(3, 4);
            kitti_pose.block<3, 3>(0, 0) = r_matrix.toRotationMatrix();
            kitti_pose(0, 3) = p.pose.position.x;
            kitti_pose(1, 3) = p.pose.position.y;
            kitti_pose(2, 3) = p.pose.position.z;
            ofs << std::fixed << std::setprecision(18) << kitti_pose(0, 0) << " " << kitti_pose(0, 1) << " "
                << kitti_pose(0, 2) << " " << kitti_pose(0, 3) << " " << kitti_pose(1, 0) << " " << kitti_pose(1, 1) << " "
                << kitti_pose(1, 2) << " " << kitti_pose(1, 3) << " " << kitti_pose(2, 0) << " " << kitti_pose(2, 1) << " "
                << kitti_pose(2, 2) << " " << kitti_pose(2, 3) << std::endl;
        }
        ofs.close();
    }

    ///////////////////////////  private method /////////////////////////////////////////////////////////////////////
    template <typename T>
    void LaserMapping::SetPosestamp(T &out)
    {
        out.pose.position.x = state_point_.pos(0);
        out.pose.position.y = state_point_.pos(1);
        out.pose.position.z = state_point_.pos(2);
        out.pose.orientation.x = state_point_.rot.coeffs()[0];
        out.pose.orientation.y = state_point_.rot.coeffs()[1];
        out.pose.orientation.z = state_point_.rot.coeffs()[2];
        out.pose.orientation.w = state_point_.rot.coeffs()[3];
    }

    void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_this(state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I);
        common::V3D p_global(state_point_.rot * p_this + state_point_.pos);
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->pt_num = pi->pt_num;
        po->time = pi->time;

        common::M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRIX(p_this);

        // 分离的协方差变换
        common::M3D R_total = state_point_.rot.toRotationMatrix() * state_point_.offset_R_L_I.toRotationMatrix();

        // 位置不确定度：进行刚体变换
        po->cov_pos = R_total * pi->cov_pos * R_total.transpose();

        // 几何协方差：只进行旋转（形状特征）
        po->cov_geo = R_total * pi->cov_geo * R_total.transpose();
    }

    void LaserMapping::PointBodyToWorldPub(const PointType *pi, PointType2 *const po)
    {
        common::V3D p_body(pi->x, pi->y, pi->z);
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        po->curvature = pi->pt_num;
    }

    void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po)
    {
        common::V3D p_body(pi.x(), pi.y(), pi.z());
        common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = std::abs(po->z);
    }

    void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po)
    {
        common::V3D p_body_lidar(pi->x, pi->y, pi->z);
        common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

} // namespace g2m_lio

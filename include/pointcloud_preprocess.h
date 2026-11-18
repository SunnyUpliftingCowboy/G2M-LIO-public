/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#ifndef AKF_LIO_POINTCLOUD_PREPROCESS_H
#define AKF_LIO_POINTCLOUD_PREPROCESS_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common_lib.h"

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        std::uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                      (float, time, time)(std::uint16_t, ring, ring))
// clang-format on

namespace ouster_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace ouster_ros

// clang-format off
// 仅注册 Ouster driver 必定发布的字段 (x, y, z, intensity, t)
// ring/reflectivity/ambient/range 为可选字段, 某些 Ouster 驱动版本/配置可能不发布,
// 若注册会导致 pcl::fromROSMsg 输出 "Failed to find match for field 'xxx'" 警告
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                  (std::uint32_t, t, t)
                                  )
// clang-format on

namespace g2m_lio
{

    enum class LidarType
    {
        AVIA = 1,
        VELO32,
        OUST64
    };

    /**
     * point cloud preprocess
     * just unify the point format from livox/velodyne to PCL
     */
    class PointCloudPreprocess
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        PointCloudPreprocess() = default;
        ~PointCloudPreprocess() = default;

        /// processors
        void Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudType::Ptr &pcl_out);
        void Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudType::Ptr &pcl_out);
        void Set(LidarType lid_type, double blind, int point_filter_num);

        // accessors
        double &Max_range() { return max_range_; }
        bool &EnableMaxRangeFilter() { return enable_max_range_filter_; }
        double &Blind() { return blind_; }
        int &NumScans() { return num_scans_; }
        int &PointFilterNum() { return point_filter_num_; }
        float &TimeScale() { return time_scale_; }
        LidarType GetLidarType() const { return lidar_type_; }
        void SetLidarType(LidarType lt) { lidar_type_ = lt; }
        bool deskew_ = true;

    private:
        void AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
        void Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void Oust128Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg);

        PointCloudType cloud_full_, cloud_out_;

        LidarType lidar_type_ = LidarType::AVIA;
        int point_filter_num_ = 1;
        int num_scans_ = 6;
        double blind_ = 0.01;
        double max_range_ = 500.0;
        // Whether to discard points farther than max_range_. Blind-range filtering remains active.
        bool enable_max_range_filter_ = true;
        float time_scale_ = 1e-3;
        bool given_offset_time_ = false;
    };
} // namespace g2m_lio

#endif // AKF_LIO_POINTCLOUD_PREPROCESS_H

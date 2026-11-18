/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace g2m_lio
{

    void PointCloudPreprocess::Set(LidarType lid_type, double blind, int point_filter_num)
    {
        lidar_type_ = lid_type;
        blind_ = blind;
        point_filter_num_ = point_filter_num;
    }

    void PointCloudPreprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudType::Ptr &pcl_out)
    {
        AviaHandler(msg);
        *pcl_out = cloud_out_;
    }

    void PointCloudPreprocess::Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudType::Ptr &pcl_out)
    {
        switch (lidar_type_)
        {
        case LidarType::OUST64:
            Oust128Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
        }
        *pcl_out = cloud_out_;
    }

    void PointCloudPreprocess::AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();
        int plsize = msg->point_num;

        cloud_out_.reserve(plsize);
        cloud_full_.resize(plsize);

        std::vector<bool> is_valid_pt(plsize, false);
        std::vector<uint> index(plsize);
        for (uint i = 0; i < plsize; ++i)
        {
            index[i] = i;
        }

        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i)
                      {
        if ((msg->points[i].line < num_scans_) && msg->points[i].reflectivity > -1 &&
            ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = msg->points[i].x;
                cloud_full_[i].y = msg->points[i].y;
                cloud_full_[i].z = msg->points[i].z;
                cloud_full_[i].intensity = msg->points[i].reflectivity;
                cloud_full_[i].normal_y=msg->points[i].line;
                cloud_full_[i].normal_x=i/num_scans_;
                cloud_full_[i].curvature =
                    msg->points[i].offset_time /
                    float(1000000);  // use curvature as time of each laser points, curvature unit: ms


double range = cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
               cloud_full_[i].z * cloud_full_[i].z;
                if (((abs(cloud_full_[i].x) > 1e-7) ||
                    (abs(cloud_full_[i].y) > 1e-7) ||
                    (abs(cloud_full_[i].z) > 1e-7)) &&
                    (range > (blind_ * blind_)) &&
                    (!enable_max_range_filter_ || range < (max_range_ * max_range_))) {
                    is_valid_pt[i] = true;
                }
            }
        } });

        for (uint i = 1; i < plsize; i++)
        {
            if (is_valid_pt[i])
            {
                cloud_out_.points.push_back(cloud_full_[i]);
            }
        }
    }

    void PointCloudPreprocess::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();
        pcl::PointCloud<ouster_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.size();
        cloud_out_.reserve(plsize);

        for (int i = 0; i < pl_orig.points.size(); i++)
        {
            if (i % point_filter_num_ != 0)
                continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;

            if (range < (blind_ * blind_) ||
                (enable_max_range_filter_ && range > max_range_ * max_range_))
                continue;

            Eigen::Vector3d pt_vec;
            PointType2 added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.normal_y = pl_orig.points[i].ring;
            added_pt.curvature = pl_orig.points[i].t / 1e6; // curvature unit: ms

            cloud_out_.points.push_back(added_pt);
        }
    }

    void PointCloudPreprocess::Oust128Handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();
        pcl::PointCloud<ouster_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.size();
        cloud_out_.reserve(plsize);

        // lidar_sensor_z_offset: LiDAR安装高度偏移 (m)
        // 当前设为0，若需使用球面投影去除地面时，需要先标定此参数
        double lidar_sensor_z_offset = 0.0;

        pcl::uint64_t max_time = 0;
        for (int i = 0; i < pl_orig.points.size(); i++)
        {
            if (i % point_filter_num_ != 0)
                continue;

            // 过滤NaN点
            if (std::isnan(pl_orig.points[i].x) || std::isnan(pl_orig.points[i].y) ||
                std::isnan(pl_orig.points[i].z))
                continue;

            double range = pl_orig.points[i].getVector3fMap().norm();

            if (range < blind_ || (enable_max_range_filter_ && range > max_range_))
                continue;

            PointType2 added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z - lidar_sensor_z_offset;
            added_pt.intensity = pl_orig.points[i].intensity;

            // 保存原始点索引和距离信息（用于后续处理）
            added_pt.normal_x = i;
            added_pt.normal_y = range;
            added_pt.normal_z = 0;
            added_pt.curvature = pl_orig.points[i].t / 1e6; // curvature unit: ms

            cloud_out_.points.push_back(added_pt);

            if (pl_orig.points[i].t > max_time)
                max_time = pl_orig.points[i].t;
        }
        cloud_out_.header.stamp = max_time;
    }

    void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();

        pcl::PointCloud<velodyne_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.points.size();
        cloud_out_.reserve(plsize);

        /*** These variables only works when no point timestamps given ***/
        double omega_l = 3.61; // scan angular velocity
        std::vector<bool> is_first(num_scans_, true);
        std::vector<double> yaw_fp(num_scans_, 0.0);   // yaw of first scan point
        std::vector<float> yaw_last(num_scans_, 0.0);  // yaw of last scan point
        std::vector<float> time_last(num_scans_, 0.0); // last offset time
        /*****************************************************************/

        if (!deskew_ || pl_orig.points[plsize - 1].time > 0)
        {
            given_offset_time_ = true;
        }
        else
        {
            given_offset_time_ = false;
            double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
            double yaw_end = yaw_first;
            int layer_first = pl_orig.points[0].ring;
            for (uint i = plsize - 1; i > 0; i--)
            {
                if (pl_orig.points[i].ring == layer_first)
                {
                    yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                    break;
                }
            }
        }

        for (int i = 0; i < plsize; i++)
        {
            if (pl_orig.points[i].z < -5)
                continue;
            PointType2 added_pt;

            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;

            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.normal_y = pl_orig.points[i].ring;
            added_pt.curvature = pl_orig.points[i].time * time_scale_; // curvature unit: ms

            if (!given_offset_time_)
            {
                int layer = pl_orig.points[i].ring;
                double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

                if (is_first[layer])
                {
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    added_pt.curvature = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt.curvature;
                    continue;
                }

                // compute offset time
                if (yaw_angle <= yaw_fp[layer])
                {
                    added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
                }
                else
                {
                    added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
                }

                if (added_pt.curvature < time_last[layer])
                    added_pt.curvature += 360.0 / omega_l;

                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
            }

            if (i % point_filter_num_ == 0)
            {
                double range = sqrt(added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z);
                if (range > blind_ && (!enable_max_range_filter_ || range < max_range_))
                {
                    cloud_out_.points.push_back(added_pt);
                }
            }
        }
    }
} // namespace g2m_lio

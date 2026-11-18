/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>

#include "laser_mapping.h"

/// run the lidar mapping in online mode

DEFINE_string(traj_log_file, (std::string(std::string(ROOT_DIR) + "Log/" + "traj.txt")), "path to traj log file");
void SigHandle(int sig)
{
    g2m_lio::options::FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv)
{
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    ros::init(argc, argv, "akf_lio");
    ros::NodeHandle nh;

    auto laser_mapping = std::make_shared<g2m_lio::LaserMapping>();
    laser_mapping->InitROS(nh);

    signal(SIGINT, SigHandle);
    signal(SIGTERM, SigHandle);
    // Bag playback stops publishing /clock at EOF. Keep the polling loop and
    // signal-driven trajectory saving responsive even when ROS time is frozen.
    ros::WallRate rate(5000);

    // online, almost same with offline, just receive the messages from ros
    while (ros::ok())
    {
        if (g2m_lio::options::FLAG_EXIT)
        {
            break;
        }
        ros::spinOnce();
        laser_mapping->Run();
        rate.sleep();
    }

    g2m_lio::Timer::PrintAll();
    LOG(INFO) << "save trajectory to: " << FLAGS_traj_log_file;
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    return 0;
}

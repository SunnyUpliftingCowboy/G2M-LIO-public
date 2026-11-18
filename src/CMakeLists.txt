# G2M-LIO modification notice (2026-10-28).
# This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
# This dated notice was added on the date above; earlier modification dates
# were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
# Existing upstream copyright and license notices are retained.

add_library(${PROJECT_NAME}
        laser_mapping.cc
        pointcloud_preprocess.cc
        options.cc
        utils.cc
        covariance_propagation.cc
        )

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp livox_ros_driver_gencpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${OpenCV_LIBRARIES}
        tbb
        glog
        yaml-cpp
        )



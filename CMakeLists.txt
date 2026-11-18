# G2M-LIO modification notice (2026-10-28).
# This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
# This dated notice was added on the date above; earlier modification dates
# were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
# Existing upstream copyright and license notices are retained.

cmake_minimum_required(VERSION 3.5)
project(g2m_lio)

# Suppress CMake warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS_OVERRIDE ON)

# Suppress deprecation warnings
set(CMAKE_WARN_DEPRECATED OFF)

# Suppress specific CMake warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS_OVERRIDE ON)

# Suppress googletest warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

# Suppress compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb -w")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
add_subdirectory(thirdparty/livox_ros_driver)

include(cmake/packages.cmake)

#definitions
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)

if (WITH_IVOX_NODE_TYPE_PHC)
    message("USING_IVOX_NODE_TYPE_PHC")
    add_definitions(-DIVOX_NODE_TYPE_PHC)
else ()
    message("USING_IVOX_NODE_TYPE_DEFAULT")
endif()

add_subdirectory(src)
add_subdirectory(app)

if(CATKIN_ENABLE_TESTING)
    add_executable(imu_boundaries_test EXCLUDE_FROM_ALL tests/imu_boundaries_test.cc)
    add_dependencies(imu_boundaries_test ${PROJECT_NAME}_gencpp)
    target_link_libraries(imu_boundaries_test ${catkin_LIBRARIES} glog)
    add_test(NAME imu_boundaries COMMAND imu_boundaries_test)
endif()

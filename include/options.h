/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */


#ifndef AKF_LIO_OPTIONS_H
#define AKF_LIO_OPTIONS_H

namespace g2m_lio::options
{

  /// fixed params
  constexpr double INIT_TIME = 0.1;
  constexpr double LASER_POINT_COV = 0.001;
  constexpr int PUBFRAME_PERIOD = 20;
  constexpr int NUM_MATCH_POINTS = 100;   // required matched points in current
  constexpr int MIN_NUM_MATCH_POINTS = 5; // minimum matched points in current

  /// configurable params
  extern int NUM_MAX_ITERATIONS; // max iterations of ekf
  extern int MAX_FEA_NUM;
  extern double TIME_TO_DELETE_LOCAL_MAP;
  extern double init_free_ratio;
  extern float T_MAL; // mal threshold
  extern float LIDAR_COV;
  extern bool FLAG_EXIT; // flag for exitting

  // Point-to-point ICP parameters
  extern double tau_planar;      // Planarity threshold (default: 0.2)
  extern bool enable_p2p_icp;    // Enable point-to-point ICP (default: true)

} // namespace g2m_lio::options

#endif // AKF_LIO_OPTIONS_H

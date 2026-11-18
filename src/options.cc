/*
 * G2M-LIO modification notice (2026-10-28).
 * This file contains G2M-LIO adaptations of AKF-LIO (45fc811ba6fe).
 * This dated notice was added on the date above; earlier modification dates
 * were not reconstructed. See THIRD_PARTY_NOTICES.md at the repository root.
 * Existing upstream copyright and license notices are retained.
 */

#include "options.h"

namespace g2m_lio::options
{
  float LIDAR_COV = 0.001;
  int NUM_MAX_ITERATIONS = 0;
  int MAX_FEA_NUM = 10000;
  double TIME_TO_DELETE_LOCAL_MAP = 1000;
  double init_free_ratio = 0.5;
  float T_MAL = 3; // mal threshold
  bool FLAG_EXIT = false;

  // Point-to-point ICP parameters
  double tau_planar = 0.2;      // Planarity threshold
  bool enable_p2p_icp = true;    // Enable point-to-point ICP

} // namespace g2m_lio::options
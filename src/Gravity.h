#pragma once

#include "SimInfo.h"

namespace fv2d
{

/**
 * @brief Computes the analytical gravity at a certain position for a certain direction
 *
 * @param i, j indices of the current cell
 * @param dir direction of the gravity to return
 * @param params the parameters of the run
 */
KOKKOS_INLINE_FUNCTION
float getAnalyticalGravity(int i, int j, IDir dir, const DeviceParams &params)
{
  Pos pos = getPos(params, i, j);
  real_t g;

  switch (params.analytical_gravity_mode)
  {
  case AGM_HOT_BUBBLE:
  default:
    g = params.hot_bubble_g0 * Kokkos::sin(pos[IY] * M_PI * 2.0 / params.ymax);
  }

  return g;
}

/**
 * @brief Method to compute the gravitational acceleration along a direction
 *
 * @param i, j indices of the current cell
 * @param dir direction of the gravity to return
 * @param params the parameters of the run
 */
KOKKOS_INLINE_FUNCTION
float getGravity(int i, int j, IDir dir, const DeviceParams &params)
{
  real_t g;
  switch (params.gravity_mode)
  {
  case GRAV_CONSTANT:
    g = (dir == IX ? params.gx : params.gy);
    break;
  case GRAV_ANALYTICAL:
    g = getAnalyticalGravity(i, j, dir, params);
    break;
  case GRAV_NONE:
  default:
    g = 0.0;
    break;
  }

  return g;
}

} // namespace fv2d

#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Orszag-Tang vortex
 */
KOKKOS_INLINE_FUNCTION
void initOrszagTang(Array Q, int i, int j, const DeviceParams &params)
{
  const real_t B0 = 1 / Kokkos::sqrt(4 * M_PI);
  Pos pos         = getPos(params, i, j);
  real_t x        = pos[IX];
  real_t y        = pos[IY];

  Q(j, i, IR)   = params.gamma0 * params.gamma0 * B0 * B0;
  Q(j, i, IU)   = -sin(2.0 * M_PI * y);
  Q(j, i, IV)   = sin(2.0 * M_PI * x);
  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = params.gamma0 * B0 * B0;
  Q(j, i, IBX)  = -B0 * sin(2.0 * M_PI * y);
  Q(j, i, IBY)  = B0 * sin(4.0 * M_PI * x);
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

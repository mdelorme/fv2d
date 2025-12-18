#pragma once

#include "../../SimInfo.h"

namespace fv2d
{
/**
 * @brief Field Advection Loop
 */
KOKKOS_INLINE_FUNCTION
void initFieldLoopAdvection(Array Q, int i, int j, const DeviceParams &params)
{
  const real_t x0 = 0.5 * (params.xmin + params.xmax);
  const real_t y0 = 0.5 * (params.ymin + params.ymax);
  Pos pos         = getPos(params, i, j);
  real_t xi       = x0 - pos[IX];
  real_t yj       = y0 - pos[IY];
  real_t r        = sqrt(xi * xi + yj * yj);
  const real_t r0 = 0.3, A0 = 0.001;

  Q(j, i, IR)   = 1.0;
  Q(j, i, IU)   = 2.0;
  Q(j, i, IV)   = 1.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = 1.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;

  if (r < r0)
  {
    Q(j, i, IBX) = -pos[IY] * A0 / r;
    Q(j, i, IBY) = pos[IX] * A0 / r;
  }
  else
  {
    Q(j, i, IBX) = 0.0;
    Q(j, i, IBY) = 0.0;
  }
}

} // namespace fv2d

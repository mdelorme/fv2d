#pragma once

#include "../../SimInfo.h"

namespace fv2d

{

/**
 * @brief MHD Rotor Test
 */
KOKKOS_INLINE_FUNCTION
void initMHDRotor(Array Q, int i, int j, const DeviceParams &params)
{
  const real_t x0 = 0.5 * (params.xmin + params.xmax);
  const real_t y0 = 0.5 * (params.ymin + params.ymax);
  Pos pos         = getPos(params, i, j);
  real_t xi       = x0 - pos[IX];
  real_t yj       = y0 - pos[IY];
  real_t r        = sqrt(xi * xi + yj * yj);
  const real_t r0 = 0.1, r1 = 0.115, u0 = 2.0;
  const real_t f  = (r1 - r) / (r1 - r0);
  const real_t B0 = 1.0 / sqrt(4 * M_PI);

  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = 1.0;
  Q(j, i, IBX)  = 5 * B0;
  Q(j, i, IBY)  = 0.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;

  if (r < r0)
  {
    Q(j, i, IR) = 10.0;
    Q(j, i, IU) = (u0 / r0) * (0.5 - pos[IY]);
    Q(j, i, IV) = (u0 / r0) * (pos[IX] - 0.5);
  }
  else if (r1 < r && r <= r0)
  {
    Q(j, i, IR) = 1 + 9 * f;
    Q(j, i, IU) = (f * u0 / r0) * (0.5 - pos[IY]);
    Q(j, i, IV) = (f * u0 / r0) * (pos[IX] - 0.5);
  }
  else
  {
    Q(j, i, IR) = 1.0;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
  }
}

} // namespace fv2d

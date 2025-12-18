#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief MHD Blast Standard Configuration
 */
KOKKOS_INLINE_FUNCTION
void initBlastMHDStandard(Array Q, int i, int j, const DeviceParams &params)
{
  real_t x0 = 0.5 * (params.xmin + params.xmax);
  real_t y0 = 0.5 * (params.ymin + params.ymax);
  Pos pos   = getPos(params, i, j);

  real_t xi = x0 - pos[IX];
  real_t yj = y0 - pos[IY];
  real_t r  = sqrt(xi * xi + yj * yj);

  if (r < 0.1)
  {
    Q(j, i, IP) = 10.0;
  }
  else
  {
    Q(j, i, IP) = 0.1;
  }
  Q(j, i, IR)   = 1.0;
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 0.5 * Kokkos::sqrt(2.0);
  Q(j, i, IBY)  = 0.5 * Kokkos::sqrt(2.0);
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

/**
 * @brief MHD Blast Low Beta Configuration
 */
KOKKOS_INLINE_FUNCTION
void initBlastMHDLowBeta(Array Q, int i, int j, const DeviceParams &params)
{
  real_t x0 = 0.5 * (params.xmin + params.xmax);
  real_t y0 = 0.5 * (params.ymin + params.ymax);
  Pos pos   = getPos(params, i, j);

  real_t xi = x0 - pos[IX];
  real_t yj = y0 - pos[IY];
  real_t r  = sqrt(xi * xi + yj * yj);

  if (r < 0.1)
  {
    Q(j, i, IP) = 1000.0;
  }
  else
  {
    Q(j, i, IP) = 0.1;
  }

  Q(j, i, IR)   = 1.0;
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 250.0 / Kokkos::sqrt(2.0);
  Q(j, i, IBY)  = 250.0 / Kokkos::sqrt(2.0);
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

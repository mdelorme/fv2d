#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Dai and Woodward test
 */
KOKKOS_INLINE_FUNCTION
void initDaiWoodward(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos             = getPos(params, i, j);
  real_t x            = pos[IX];
  const real_t midbox = 0.5 * (params.xmax + params.xmin);
  const real_t B0     = 1.0 / std::sqrt(4 * M_PI);

  if (x < midbox)
  {
    Q(j, i, IR)  = 1.08;
    Q(j, i, IU)  = 1.2;
    Q(j, i, IV)  = 0.01;
    Q(j, i, IW)  = 0.5;
    Q(j, i, IP)  = 0.95;
    Q(j, i, IBY) = B0 * 3.6;
  }
  else
  {
    Q(j, i, IR)  = 1.0;
    Q(j, i, IU)  = 0.0;
    Q(j, i, IV)  = 0.0;
    Q(j, i, IW)  = 0.0;
    Q(j, i, IP)  = 1.0;
    Q(j, i, IBY) = B0 * 4.0;
  }
  Q(j, i, IBX)  = B0 * 4.0;
  Q(j, i, IBZ)  = B0 * 2.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

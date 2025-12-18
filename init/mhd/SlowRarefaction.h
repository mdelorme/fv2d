#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Slow Rarefaction Test
 */
KOKKOS_INLINE_FUNCTION
void initSlowRarefaction(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos             = getPos(params, i, j);
  real_t x            = pos[IX];
  const real_t midbox = 0.5 * (params.xmax + params.xmin);

  if (x < midbox)
  {
    Q(j, i, IR)  = 1.0;
    Q(j, i, IU)  = 0.0;
    Q(j, i, IV)  = 0.0;
    Q(j, i, IP)  = 2.0;
    Q(j, i, IBY) = 0.0;
  }
  else
  {
    Q(j, i, IR)  = 0.2;
    Q(j, i, IU)  = 1.186;
    Q(j, i, IV)  = 2.967;
    Q(j, i, IP)  = 0.1368;
    Q(j, i, IBY) = 1.6405;
  }
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 1.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

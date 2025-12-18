#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void initShuOsher(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos         = getPos(params, i, j);
  real_t x        = pos[IX];
  const real_t x0 = -4.0; // shock interface
  if (x <= x0)
  {
    Q(j, i, IR)  = 3.5;
    Q(j, i, IU)  = 5.8846;
    Q(j, i, IV)  = 1.1198;
    Q(j, i, IP)  = 42.0267;
    Q(j, i, IBY) = 3.6359;
  }
  else
  {
    Q(j, i, IR)  = 1.0 + 0.2 * Kokkos::sin(5.0 * x);
    Q(j, i, IU)  = 0.0;
    Q(j, i, IV)  = 0.0;
    Q(j, i, IP)  = 1.0;
    Q(j, i, IBY) = 1.0;
  }
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 1.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

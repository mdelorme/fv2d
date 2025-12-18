#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void initArtificialNonZeroDivB(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];

  Q(j, i, IR)   = 1.0;
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = 1.0;
  Q(j, i, IBY)  = 0.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;

  if (x <= -0.8)
    Q(j, i, IBX) = 0.0;
  else if (x <= -0.6)
    Q(j, i, IBX) = -2.0 * (x + 0.8);
  else if (x <= 0.6)
    Q(j, i, IBX) = Kokkos::exp(-0.5 * (x / 0.11) * (x / 0.11));
  else
    Q(j, i, IBX) = 0.5;
}

} // namespace fv2d

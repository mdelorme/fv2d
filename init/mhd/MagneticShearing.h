#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void shearB(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos         = getPos(params, i, j);
  const real_t d  = (pos[IY] - params.ymin) / (params.ymax - params.ymin);
  const real_t A  = Kokkos::sin(M_PI * d);
  const real_t U0 = 1.0;
  const real_t V0 = 0.0;

  Q(j, i, IR)   = 10.0;
  Q(j, i, IP)   = 10.0;
  Q(j, i, IU)   = U0 * A;
  Q(j, i, IV)   = V0 * A;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 0.0;
  Q(j, i, IBY)  = 1e-1;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

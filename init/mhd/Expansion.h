#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief First Expansion Test
 */
KOKKOS_INLINE_FUNCTION
void initExpansion1(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos             = getPos(params, i, j);
  real_t x            = pos[IX];
  const real_t midbox = 0.5 * (params.xmax + params.xmin);

  if (x < midbox)
  {
    Q(j, i, IU) = -3.1;
  }
  else
  {
    Q(j, i, IU) = 3.1;
  }
  Q(j, i, IR)   = 1.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = 0.45;
  Q(j, i, IBX)  = 0.0;
  Q(j, i, IBY)  = 0.5;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

/**
 * @brief Second Expansion Test
 */
KOKKOS_INLINE_FUNCTION
void initExpansion2(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos             = getPos(params, i, j);
  real_t x            = pos[IX];
  const real_t midbox = 0.5 * (params.xmax + params.xmin);

  if (x < midbox)
  {
    Q(j, i, IU) = -3.1;
  }
  else
  {
    Q(j, i, IU) = 3.1;
  }

  Q(j, i, IR)   = 1.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IP)   = 0.45;
  Q(j, i, IBX)  = 1.0;
  Q(j, i, IBY)  = 0.5;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

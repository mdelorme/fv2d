#pragma once

#include "../../SimInfo.h"

namespace fv2d
{
/**
 * @brief Brio-Wu (MHD Sod Shock) tube aligned along the X axis
 */
KOKKOS_INLINE_FUNCTION
void initBrioWu1X(Array Q, int i, int j, const DeviceParams &params)
{
  if (getPos(params, i, j)[IX] <= 0.5)
  {
    Q(j, i, IR)  = 1.0;
    Q(j, i, IP)  = 1.0;
    Q(j, i, IBY) = 1.0;
  }
  else
  {
    Q(j, i, IR)  = 0.125;
    Q(j, i, IP)  = 0.1;
    Q(j, i, IBY) = -1.0;
  }
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 0.75;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

/**
 * @brief MHD Sod Shock tube aligned along the Y axis
 */
KOKKOS_INLINE_FUNCTION
void initBrioWu1Y(Array Q, int i, int j, const DeviceParams &params)
{
  if (getPos(params, i, j)[IY] <= 0.5)
  {
    Q(j, i, IR)  = 1.0;
    Q(j, i, IP)  = 1.0;
    Q(j, i, IBX) = 1.0;
  }
  else
  {
    Q(j, i, IR)  = 0.125;
    Q(j, i, IP)  = 0.1;
    Q(j, i, IBX) = -1.0;
  }
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBY)  = 0.75;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

/**
 * @brief Second Brio-Wu test
 */
KOKKOS_INLINE_FUNCTION
void initBrioWu2(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos             = getPos(params, i, j);
  real_t x            = pos[IX];
  const real_t midbox = 0.5 * (params.xmax + params.xmin);

  if (x < midbox)
  {
    Q(j, i, IR)  = 1.0;
    Q(j, i, IP)  = 1000.0;
    Q(j, i, IBY) = 1.0;
  }
  else
  {
    Q(j, i, IR)  = 0.125;
    Q(j, i, IP)  = 0.1;
    Q(j, i, IBY) = -1.0;
  }
  Q(j, i, IU)   = 0.0;
  Q(j, i, IV)   = 0.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 0.0;
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

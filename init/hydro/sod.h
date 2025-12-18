#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Sod Shock tube aligned along the X axis
 */
KOKKOS_INLINE_FUNCTION
void initSodX(Array Q, int i, int j, const DeviceParams &params)
{
  if (getPos(params, i, j)[IX] <= 0.5)
  {
    Q(j, i, IR) = 1.0;
    Q(j, i, IP) = 1.0;
    Q(j, i, IU) = 0.0;
  }
  else
  {
    Q(j, i, IR) = 0.125;
    Q(j, i, IP) = 0.1;
    Q(j, i, IU) = 0.0;
  }
}

/**
 * @brief Sod Shock tube aligned along the Y axis
 */
KOKKOS_INLINE_FUNCTION
void initSodY(Array Q, int i, int j, const DeviceParams &params)
{
  if (getPos(params, i, j)[IY] <= 0.5)
  {
    Q(j, i, IR) = 1.0;
    Q(j, i, IP) = 1.0;
    Q(j, i, IU) = 0.0;
  }
  else
  {
    Q(j, i, IR) = 0.125;
    Q(j, i, IP) = 0.1;
    Q(j, i, IU) = 0.0;
  }
}

} // namespace fv2d

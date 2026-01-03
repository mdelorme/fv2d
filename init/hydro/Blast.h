#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Sedov blast initial conditions
 */
KOKKOS_INLINE_FUNCTION
void initBlast(Array Q, int i, int j, const DeviceParams &params)
{
  real_t xmid = 0.5 * (params.xmin + params.xmax);
  real_t ymid = 0.5 * (params.ymin + params.ymax);

  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];
  real_t y = pos[IY];

  real_t xr = xmid - x;
  real_t yr = ymid - y;
  real_t r  = sqrt(xr * xr + yr * yr);

  if (r < 0.2)
  {
    Q(j, i, IR) = 1.0;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IP) = 10.0;
  }
  else
  {
    Q(j, i, IR) = 1.2;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IP) = 0.1;
  }
}
} // namespace fv2d

#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief MHD Rotated Shock Tube. Brio and Wu Shock Tube rotated by an angle \theta
 */
KOKKOS_INLINE_FUNCTION
void initRotatedShockTube(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos      = getPos(params, i, j);
  real_t theta = Kokkos::atan(-2.0);

  real_t xt = tan(theta) * (pos[IX] - 0.5);
  real_t yt = (pos[IY] - 0.5);
  real_t B0 = 1.0 / sqrt(4.0 * M_PI);

  Q(j, i, IR)   = 1.0;
  Q(j, i, IW)   = 0.0;
  Q(j, i, IBX)  = 5 * B0 * (cos(theta) + sin(theta));
  Q(j, i, IBY)  = 5 * B0 * (cos(theta) - sin(theta));
  Q(j, i, IBZ)  = 0.0;
  Q(j, i, IPSI) = 0.0;

  if (xt < yt)
  {
    Q(j, i, IU) = 10.0 * cos(theta);
    Q(j, i, IV) = -10.0 * sin(theta);
    Q(j, i, IP) = 20.0;
  }
  else
  {
    Q(j, i, IU) = -10.0 * cos(theta);
    Q(j, i, IV) = 10.0 * sin(theta);
    Q(j, i, IP) = 1.0;
  }
}

} // namespace fv2d

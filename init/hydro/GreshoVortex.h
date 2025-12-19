#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Gresho-Vortex setup for Low-mach flows
 *
 * Based on Miczek et al. 2015 "New numerical solver for flows at various Mach numbers"
 */
KOKKOS_INLINE_FUNCTION
void initGreshoVortex(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos           = getPos(params, i, j);
  const real_t xmid = 0.5 * (params.xmin + params.xmax);
  const real_t ymid = 0.5 * (params.ymin + params.ymax);
  const real_t xr   = pos[IX] - xmid;
  const real_t yr   = pos[IY] - ymid;
  const real_t r    = sqrt(xr * xr + yr * yr);

  // Pressure is given from density and Mach
  const real_t p0 = params.gresho_density / (params.gamma0 * params.gresho_Mach * params.gresho_Mach);

  Q(j, i, IR) = params.gresho_density;

  real_t u_phi;
  if (r < 0.2)
  {
    u_phi       = 5.0 * r;
    Q(j, i, IP) = p0 + 12.5 * r * r;
  }
  else if (r < 0.4)
  {
    u_phi       = 2.0 - 5.0 * r;
    Q(j, i, IP) = p0 + 12.5 * r * r + 4.0 * (1.0 - 5.0 * r + log(5.0 * r));
  }
  else
  {
    u_phi       = 0.0;
    Q(j, i, IP) = p0 - 2.0 + 4.0 * log(2.0);
  }

  const real_t xnr = xr / r;
  const real_t ynr = yr / r;
  Q(j, i, IU)      = -ynr * u_phi;
  Q(j, i, IV)      = xnr * u_phi;
}

} // namespace fv2d

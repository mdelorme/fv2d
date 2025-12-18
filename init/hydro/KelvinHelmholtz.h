#pragma once

#include "../../SimInfo.h"

namespace fv2d
{

/**
 * @brief Kelvin-Helmholtz instability setup
 *
 * Taken from Lecoanet et al, "A validated non-linear Kelvin–Helmholtz benchmark for numerical hydrodynamics"
 * 2016, MNRAS
 */
KOKKOS_INLINE_FUNCTION
void initKelvinHelmholtz(Array Q, int i, int j, const DeviceParams &params)
{
  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];
  real_t y = pos[IY];

  const real_t q1  = Kokkos::tanh((y - params.kh_y1) / params.kh_a);
  const real_t q2  = Kokkos::tanh((y - params.kh_y2) / params.kh_a);
  const real_t s2  = params.kh_sigma * params.kh_sigma;
  const real_t dy1 = (y - params.kh_y1) * (y - params.kh_y1);
  const real_t dy2 = (y - params.kh_y2) * (y - params.kh_y2);
  const real_t rho = 1.0 + params.kh_rho_fac * 0.5 * (q1 - q2);
  const real_t u   = params.kh_uflow * (q1 - q2 - 1.0);
  const real_t v   = params.kh_amp * Kokkos::sin(2.0 * M_PI * x) * (Kokkos::exp(-dy1 / s2) + Kokkos::exp(-dy2 / s2));

  Q(j, i, IR) = rho;
  Q(j, i, IU) = u;
  Q(j, i, IV) = v;
  Q(j, i, IP) = params.kh_P0;
}

} // namespace fv2d

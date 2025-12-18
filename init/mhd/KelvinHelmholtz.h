#pragma once

#include <Kokkos_Random.hpp>
#include "../../SimInfo.h"

using RandomPool = Kokkos::Random_XorShift64_Pool<>;

namespace fv2d
{
/**
 * @brief Kelvin-Helmholtz instability
 */
KOKKOS_INLINE_FUNCTION
void initMHDKelvinHelmholtz(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool)
{
  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];
  real_t y = pos[IY];

  if (Kokkos::abs(y) <= 0.25)
  {
    Q(j, i, IR) = 2.0;
    Q(j, i, IU) = 0.5;
  }
  else
  {
    Q(j, i, IR) = 1.0;
    Q(j, i, IU) = -0.5;
  }

  Q(j, i, IV)  = 0.0;
  Q(j, i, IW)  = 0.0;
  Q(j, i, IP)  = 2.5;
  Q(j, i, IBX) = 0.5 / Kokkos::sqrt(4 * M_PI);
  Q(j, i, IBY) = 0.0;
  Q(j, i, IBZ) = 0.0;

  // Add some perturbation on both the x and y components of the velocity
  // We take a 0.01 peak-to-peak amplitude
  auto generator = random_pool.get_state();
  real_t pert_vx = generator.drand(-0.05, 0.05);
  real_t pert_vy = generator.drand(-0.05, 0.05);
  random_pool.free_state(generator);

  Q(j, i, IU) += pert_vx;
  Q(j, i, IV) += pert_vy;
  Q(j, i, IPSI) = 0.0;
}

} // namespace fv2d

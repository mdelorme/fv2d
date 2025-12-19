#pragma once

#include <Kokkos_Random.hpp>
#include "../../SimInfo.h"

using RandomPool = Kokkos::Random_XorShift64_Pool<>;

namespace fv2d
{

/**
 * @brief Stratified convection based on Cattaneo et al. 1991
 */
KOKKOS_INLINE_FUNCTION
void initC91(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool)
{
  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];
  real_t y = pos[IY];

  real_t T   = (1.0 + params.theta1 * y);
  real_t rho = pow(T, params.m1);
  real_t prs = pow(T, params.m1 + 1.0);

  auto generator = random_pool.get_state();
  real_t pert    = params.c91_pert * (generator.drand(-0.5, 0.5));
  random_pool.free_state(generator);

  prs = prs * (1.0 + pert);

  Q(j, i, IR) = rho;
  Q(j, i, IU) = 0.0;
  Q(j, i, IV) = 0.0;
  Q(j, i, IP) = prs;
}

} // namespace fv2d

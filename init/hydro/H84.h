#pragma once

#include "../../SimInfo.h"
#include "../InitData.h"

namespace fv2d
{
/**
 * @brief Stratified convection based on Hurlburt et al 1984
 */
KOKKOS_INLINE_FUNCTION
void initH84(Array Q, int i, int j, const DeviceParams &params, const InitData &ini_data)
{
  Pos pos  = getPos(params, i, j);
  real_t x = pos[IX];
  real_t y = pos[IY];

  real_t rho = pow(y, params.m1);
  real_t prs = pow(y, params.m1 + 1.0);

  auto generator = ini_data.random_pool.get_state();
  real_t pert    = params.h84_pert * (generator.drand(-0.5, 0.5));
  ini_data.random_pool.free_state(generator);

  Q(j, i, IR) = rho;
  Q(j, i, IU) = 0.0;
  Q(j, i, IV) = pert;
  Q(j, i, IP) = prs;
}

} // namespace fv2d

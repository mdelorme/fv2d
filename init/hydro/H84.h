#pragma once

#include "../InitFactory.h"

namespace fv2d
{
/**
 * @brief Stratified convection based on Hurlburt et al 1984
 */
struct InitH84 : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
      "Initialization",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        Pos pos  = getPos(params, i, j);
        real_t x = pos[IX];
        real_t y = pos[IY];

        real_t rho = pow(y, params.m1);
        real_t prs = pow(y, params.m1 + 1.0);

        auto generator = init_data.random_pool.get_state();
        real_t pert    = params.h84_pert * (generator.drand(-0.5, 0.5));
        init_data.random_pool.free_state(generator);

        Q(j, i, IR) = rho;
        Q(j, i, IU) = 0.0;
        Q(j, i, IV) = pert;
        Q(j, i, IP) = prs;
      });
  }
};

REGISTER_INIT(InitH84, H84)

} // namespace fv2d

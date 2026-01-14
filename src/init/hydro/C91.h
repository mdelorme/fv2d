#pragma once

#include "../InitFactory.h"

namespace fv2d
{

/**
 * @brief Stratified convection based on Cattaneo et al. 1991
 */
struct InitC91 : public InitFormula
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

          real_t T   = (1.0 + params.theta1 * y);
          real_t rho = pow(T, params.m1);
          real_t prs = pow(T, params.m1 + 1.0);

          auto generator = init_data.random_pool.get_state();
          real_t pert    = params.c91_pert * (generator.drand(-0.5, 0.5));
          init_data.random_pool.free_state(generator);

          prs = prs * (1.0 + pert);

          Q(j, i, IR) = rho;
          Q(j, i, IU) = 0.0;
          Q(j, i, IV) = 0.0;
          Q(j, i, IP) = prs;
        });
  }
};

REGISTER_INIT(InitC91, C91)

} // namespace fv2d

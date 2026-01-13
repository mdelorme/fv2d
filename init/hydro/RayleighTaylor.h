#pragma once

#include "../InitFormula.h"

namespace fv2d
{

/**
 * @brief Rayleigh-Taylor instability setup
 */
struct InitRayleighTaylor : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
      "Initialization",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        real_t ymid = 0.5 * (params.ymin + params.ymax);

        Pos pos  = getPos(params, i, j);
        real_t x = pos[IX];
        real_t y = pos[IY];

        const real_t P0 = 2.5;

        if (y < ymid)
        {
          Q(j, i, IR) = 1.0;
          Q(j, i, IU) = 0.0;
          Q(j, i, IP) = P0 + 0.1 * params.gy * y;
        }
        else
        {
          Q(j, i, IR) = 2.0;
          Q(j, i, IU) = 0.0;
          Q(j, i, IP) = P0 + 0.1 * params.gy * y;
        }

        if (y > -1.0 / 3.0 && y < 1.0 / 3.0)
          Q(j, i, IV) = 0.01 * (1.0 + cos(4 * M_PI * x)) * (1 + cos(3.0 * M_PI * y)) / 4.0;
      });
  }
};

} // namespace fv2d

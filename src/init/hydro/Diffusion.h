#pragma once

#include "../InitFactory.h"

namespace fv2d
{

/**
 * @brief Simple diffusion test with a structure being advected on the grid
 */
struct InitDiffusion : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
        "Initialization",
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          real_t xmid = 0.5 * (params.xmin + params.xmax);
          real_t ymid = 0.5 * (params.ymin + params.ymax);

          Pos pos = getPos(params, i, j);

          real_t x0 = (pos[IX] - xmid);
          real_t y0 = (pos[IY] - ymid);

          real_t r = sqrt(x0 * x0 + y0 * y0);

          if (r < 0.2)
            Q(j, i, IR) = 1.0;
          else
            Q(j, i, IR) = 0.1;

          Q(j, i, IP) = 1.0;
          Q(j, i, IU) = 1.0;
          Q(j, i, IV) = 1.0;
        });
  }
};

REGISTER_INIT(InitDiffusion, diffusion)

} // namespace fv2d

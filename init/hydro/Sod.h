#pragma once

#include "../InitFactory.h"

namespace fv2d {

/**
 * @todo Merge both sods into one and use a parameter instead
 */

/**
 * @brief Sod Shock tube aligned along the X axis
 * @tparam dir direction orthogonal to the shock
 */
template<IDir dir>
struct InitSod : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
      "Initialization",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        if (getPos(params, i, j)[dir] <= 0.5)
        {
          Q(j, i, IR) = 1.0;
          Q(j, i, IP) = 1.0;
          Q(j, i, IU) = 0.0;
        }
        else
        {
          Q(j, i, IR) = 0.125;
          Q(j, i, IP) = 0.1;
          Q(j, i, IU) = 0.0;
        }
      });
  }
};

REGISTER_INIT(InitSod<IX>, sod_x)
REGISTER_INIT(InitSod<IY>, sod_y)

} // namespace fv2d

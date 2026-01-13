#pragma once

#include "../InitFormula.h"

namespace fv2d {

/**
 * @todo Merge both sods into one and use a parameter instead
 */

/**
 * @brief Sod Shock tube aligned along the X axis
 */
struct InitSodX : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
      "Initialization",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        if (getPos(params, i, j)[IX] <= 0.5)
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

/**
 * @brief Sod Shock tube aligned along the Y axis
 */
struct InitSodY : public InitFormula
{
  void init(Array Q, const Params &full_params)
  {
    auto params = full_params.device_params;
    InitData init_data{full_params};

    Kokkos::parallel_for(
      "Initialization",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        if (getPos(params, i, j)[IY] <= 0.5)
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

} // namespace fv2d

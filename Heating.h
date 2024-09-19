#pragma once

#include "SimInfo.h"
#include "polyfit.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t compute_heating(real_t y, const Params &params) {
  real_t res;
  switch (params.heating_mode) {
    case HM_POLYFIT:
    {
      res = get_heating(y);
      // if (y > 0.1) res = 0.1;
      // if (y < 0) res = 0; // left ghost, otherwise it makes kappa null on the boundaries and so T cannot be fixed.
      break;
    }
    default:
      res = 0;
  }
  return res;
}

class HeatingFunctor {
public:
  Params params;

  HeatingFunctor(const Params &params) 
    : params(params) {};
  ~HeatingFunctor() = default;

  void applyHeating(Array Q, Array Unew, real_t dt) {
    auto params = this->params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    Kokkos::parallel_for(
      "Heating", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        Pos pos = getPos(params, i, j);
        real_t x = pos[IX];
        real_t y = pos[IY];

        real_t h = compute_heating(y, params);
        real_t rho = Q(j, i, IR);

        Unew(j, i, IE) += dt * rho * h;
      });
  }
};

}
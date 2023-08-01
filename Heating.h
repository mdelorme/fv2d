#pragma once

#include "SimInfo.h"

namespace fv2d {

namespace {

KOKKOS_INLINE_FUNCTION
real_t heatC2020(Array Q, const int i, const int j, const Params &params) {
  Pos pos = getPos(params, i, j);
  real_t y = pos[IY];

  real_t F = params.kappa*params.theta1*params.gamma0/(params.gamma0-1.0) * 2.0;

  const real_t H = params.c20_H;
  real_t qc;
  if (y < H) {
    qc = -1.0 - cos(2.0*M_PI*(y-H*0.5)/H);
  }
  else if (y > params.ymax-H) {
    qc = 1.0 + cos(2.0*M_PI*(params.ymax-y+H*0.5)/H);
  }
  else 
    qc = 0.0;

  return F * qc;
}

}

class HeatingFunctor {
public:
  Params params;

  HeatingFunctor(const Params &params) 
    : params(params) {};
  ~HeatingFunctor() = default;

  void applyHeating(Array Q, Array Unew, real_t dt, int ite) {
    auto params = this->params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    real_t total_heating = 0.0;
    Kokkos::parallel_reduce(
      "Heating",
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, real_t &total_heating) {
        real_t q;

        switch(params.heating_mode) {
          case HM_C2020: q = heatC2020(Q, i, j, params); break;
        } 

        // Explicit update
        Unew(j, i, IE) += dt * q;

        total_heating += dt * q;
      }, Kokkos::Sum<real_t>(total_heating));
    
    if (params.log_energy_contributions && ite % params.log_energy_frequency == 0)
      std::cout << "Total heating contribution to energy : " << total_heating << std::endl;
  }
};

}
#pragma once

#include "SimInfo.h"

namespace fv2d {

namespace {

KOKKOS_INLINE_FUNCTION
real_t cooling_layer(Array Q, const int i, const int j, const Params &params) {
  Pos pos = getPos(params, i, j);
  real_t y = pos[IY];
  real_t kappa = params.kappa*params.iso3_k2; //*params.gamma0/(params.gamma0-1.0);
  real_t F = params.iso3_theta2*kappa/params.dy;

  real_t qc = 0.0;
  real_t ydiff = y-params.iso3_dy0;
  if (fabs(ydiff) <= params.dy && ydiff < 0.0)
    qc = -F;

  return qc;
}

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

KOKKOS_INLINE_FUNCTION
real_t heatC2020_tri(Array Q, const int i, const int j, const Params &params) {
  Pos pos = getPos(params, i, j);
  real_t y = pos[IY];

  real_t F = params.kappa*params.theta1*params.gamma0/(params.gamma0-1.0) * params.c20_heating_fac;

  // Limits of the convection zone
  const real_t y1 = params.tri_y1;
  const real_t y2 = params.tri_y2;

  const real_t H = params.c20_H;
  real_t qc;
  if (y >= y1 - 0.5*H && y <= y1 + 0.5*H)
    qc = -1.0 - cos(2.0*M_PI*(y-y1)/H);
  else if (y >= y2 - 0.5*H && y <= y2 + 0.5*H)
    qc = 1.0 + cos(2.0*M_PI*(y-y2)/H);
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

    real_t total_heating = 0.0;
    Kokkos::parallel_reduce(
      "Heating",
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, real_t &total_heating) {
        real_t q;

        switch(params.heating_mode) {
          case HM_C2020: q = heatC2020(Q, i, j, params); break;
          case HM_C2020_TRI: q = heatC2020_tri(Q, i, j, params); break;
          case HM_COOLING_ISO: q = cooling_layer(Q, i, j, params); break;
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

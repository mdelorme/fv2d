#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

class ComputeDtFunctor {
public:
  Params params;

  ComputeDtFunctor(const Params &params)
    : params(params) {};
  ~ComputeDtFunctor() = default;

  real_t computeDt(Array Q, real_t max_dt, real_t t, bool diag) const {
    auto params = this->params;

    real_t inv_dt_hyp = 0.0;
    real_t inv_dt_par_tc = 0.0;
    real_t inv_dt_par_visc = 0.0;

    Kokkos::parallel_reduce("Computing DT",
                            params.range_dom,
                            KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
                              // Hydro time-step
                              State q = getStateFromArray(Q, i, j);
                              real_t cs = speedOfSound(q, params);
                              Pos pos = getPos(params, i, j);
                              real_t x = pos[IX];
                              real_t y = pos[IY];

                              real_t inv_dt_hyp_loc = (cs + fabs(q[IU]))/params.dx + (cs + fabs(q[IV]))/params.dx;

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active)
                                inv_dt_par_tc_loc = fmax(2.0*computeKappa(x, y, params) / (params.dx*params.dx) * (params.gamma0 - 1) / q[IR],
                                                         2.0*computeKappa(x, y, params) / (params.dy*params.dy) * (params.gamma0 - 1) / q[IR]);

                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active)
                                inv_dt_par_visc_loc = fmax(2.0*computeMu(x, y, params) / (params.dx*params.dx),
                                                           2.0*computeMu(x, y, params) / (params.dy*params.dy));

                              inv_dt_hyp      = fmax(inv_dt_hyp, inv_dt_hyp_loc);
                              inv_dt_par_tc   = fmax(inv_dt_par_tc, inv_dt_par_tc_loc);
                              inv_dt_par_visc = fmax(inv_dt_par_visc, inv_dt_par_visc_loc);
                            }, Kokkos::Max<real_t>(inv_dt_hyp), 
                               Kokkos::Max<real_t>(inv_dt_par_tc), 
                               Kokkos::Max<real_t>(inv_dt_par_visc));
  
    if (diag) {
      std::cout << "Computing dts at (t=" << t << ") : dt_hyp=" << 1.0/inv_dt_hyp
                << "; dt_TC="   << 1.0/inv_dt_par_tc
                << "; dt_visc=" << 1.0/inv_dt_par_visc << std::endl; 
    }

    return params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

}
#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

class ComputeDtFunctor {
public:
  Params full_params;

  ComputeDtFunctor(const Params &full_params)
    : full_params(full_params) {};
  ~ComputeDtFunctor() = default;

  real_t computeDt(Array Q, real_t max_dt, real_t t, bool diag) const {
    auto params = full_params.device_params;

    real_t inv_dt_hyp = 0.0;
    real_t inv_dt_par_tc = 0.0;
    real_t inv_dt_par_visc = 0.0;
    const real_t gdx = params.gy * params.dy;

    Kokkos::parallel_reduce("Computing DT",
                            full_params.range_dom,
                            KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
                              real_t inv_dt_hyp_loc;
                              // Traditional Hydro time-step
                              State q = getStateFromArray(Q, i, j);
                              real_t cs = speedOfSound(q, params);

                              inv_dt_hyp_loc = (cs + fabs(q[IU]))/params.dx + (cs + fabs(q[IV]))/params.dx;

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active)
                                inv_dt_par_tc_loc = fmax(2.0*computeKappa(i, j, params) / (params.dx*params.dx),
                                                         2.0*computeKappa(i, j, params) / (params.dy*params.dy));
                              
                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active)
                                inv_dt_par_visc_loc = fmax(2.0*computeMu(i, j, params) / (params.dx*params.dx),
                                                           2.0*computeMu(i, j, params) / (params.dy*params.dy));

                              inv_dt_hyp      = fmax(inv_dt_hyp, inv_dt_hyp_loc);
                              inv_dt_par_tc   = fmax(inv_dt_par_tc, inv_dt_par_tc_loc);
                              inv_dt_par_visc = fmax(inv_dt_par_visc, inv_dt_par_visc_loc);
                            }, Kokkos::Max<real_t>(inv_dt_hyp), 
                               Kokkos::Max<real_t>(inv_dt_par_tc), 
                               Kokkos::Max<real_t>(inv_dt_par_visc));
  
    if (diag) {
      std::cout << "Computing dts at (t=" << t << ") : dt_hyp=" << 1.0/inv_dt_hyp;
      if(params.thermal_conductivity_active)
        std::cout << "; dt_TC="   << 1.0/inv_dt_par_tc;
      if(params.viscosity_active)
        std::cout << "; dt_visc=" << 1.0/inv_dt_par_visc;
      std::cout << std::endl; 
    }

    return full_params.device_params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

}

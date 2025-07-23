#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

class ComputeDtFunctor {
public:
  Params full_params;
  Geometry geometry;

  ComputeDtFunctor(const Params &full_params)
    : full_params(full_params), 
    geometry(full_params.device_params) {};
  ~ComputeDtFunctor() = default;

  real_t computeDt(Array Q, real_t max_dt, real_t t, bool diag) const {
    auto params = full_params.device_params;
    auto &geometry = this->geometry;

    real_t inv_dt_hyp = 0.0;
    real_t inv_dt_par_tc = 0.0;
    real_t inv_dt_par_visc = 0.0;

    const real_t Cv = params.spl_rho.header.Cp - params.spl_rho.header.R;

    Kokkos::parallel_reduce("Computing DT",
                            full_params.range_dom,
                            KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
                              real_t dx = geometry.cellLength(i,j,IX);
                              real_t dy = geometry.cellLength(i,j,IY);
                              real_t dl = fmin(dx, dy);
                              const real_t r = norm(geometry.mapc2p_center(i,j));
                              
                              // Hydro time-step
                              State q = getStateFromArray(Q, i, j);
                              real_t cs = speedOfSound(q, params);
                              
                              real_t inv_dt_hyp_loc;
                              if (params.geometry_type == GEO_CARTESIAN)
                                inv_dt_hyp_loc = (cs + fabs(q[IU]))/params.dx + (cs + fabs(q[IV]))/params.dy;
                              else
                                inv_dt_hyp_loc = (cs + sqrt(q[IU]*q[IU] + q[IV]*q[IV]))/dl;

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active) {
                                const real_t kappa = params.spl_kappa(r);
                                const real_t rho = q[IR];
                                inv_dt_par_tc_loc = 2.0*kappa / (rho * Cv * dl*dl);
                              }
                              
                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active) {
                                inv_dt_par_visc_loc = 2.0*computeMu(r, params) / (dl*dl);
                              }

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

    if (inv_dt_hyp < 0 || inv_dt_par_tc < 0 || inv_dt_par_visc < 0)
      std::cout << "invalid dt. " << std::endl;

    return full_params.device_params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

}

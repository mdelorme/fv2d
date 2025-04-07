#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

class ComputeDtFunctor {
public:
  Params params;
  Geometry geometry;

  ComputeDtFunctor(const Params &params)
    : params(params), geometry(params) {};
  ~ComputeDtFunctor() = default;

  real_t computeDt(Array Q, real_t max_dt, real_t t, bool diag) const {
    // using DtArray = Kokkos::Array<real_t, 3>;
    // DtArray inv_dts {0,0,0};

    auto params = this->params;
    auto &geometry = this->geometry;

    real_t inv_dt_hyp = 0.0;
    real_t inv_dt_par_tc = 0.0;
    real_t inv_dt_par_visc = 0.0;

    Kokkos::parallel_reduce("Computing DT",
                            params.range_dom,
                            KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
                              real_t dx = geometry.cellLength(i,j,IX);
                              real_t dy = geometry.cellLength(i,j,IY);
                              real_t dl = fmin(dx, dy);
                              
                              // Hydro time-step
                              State q = getStateFromArray(Q, i, j);
                              real_t cs = speedOfSound(q, params);
                              
                              real_t inv_dt_hyp_loc;
                              if (params.geometry_type == GEO_CARTESIAN)
                                inv_dt_hyp_loc = (cs + fabs(q[IU]))/dx + (cs + fabs(q[IV]))/dy;
                              else
                                inv_dt_hyp_loc = (cs + sqrt(q[IU]*q[IU] + q[IV]*q[IV]))/dl;

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active)
                                inv_dt_par_tc_loc = fmax(2.0*computeKappa(i, j, params) / (dl*dl),
                                                         2.0*computeKappa(i, j, params) / (dl*dl));
                              
                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active)
                                inv_dt_par_visc_loc = fmax(2.0*computeMu(i, j, params) / (dl*dl),
                                                           2.0*computeMu(i, j, params) / (dl*dl));

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

    return params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

}
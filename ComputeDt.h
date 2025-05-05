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
    //using DtArray = Kokkos::Array<real_t, 3>;
    //DtArray inv_dts {0,0,0};

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
                              real_t inv_dt_hyp_loc = (cs + fabs(q[IU]))/params.dx + (cs + fabs(q[IV]))/params.dy;

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active)
                                inv_dt_par_tc_loc = fmax(2.0*computeKappa(i, j, params) / (params.dx*params.dx),
                                                         2.0*computeKappa(i, j, params) / (params.dy*params.dy));
                              
                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active)
                                inv_dt_par_visc_loc = fmax(2.0*computeMu(i, j, params) / (params.dx*params.dx),
                                                           2.0*computeMu(i, j, params) / (params.dy*params.dy));
                              #ifdef MHD
                              const real_t Bx = q[IBX];
                              const real_t By = q[IBY];
                              const real_t Bz = q[IBZ];
                              const real_t gr = cs*cs*q[IR];
                              const real_t Bt2 [] = {By*By+Bz*Bz,
                                                     Bx*Bx+Bz*Bz,
                                                     Bx*Bx+By*By};
                              const real_t B2 = Bx*Bx + By*By + Bz*Bz;
                              const real_t cf1 = gr-B2;
                              const real_t V [] = {q[IU], q[IV], q[IW]};
                              const real_t D [] = {params.dx, params.dy};
                              
                              const int ndim = 2;
                              for (int i=0; i < ndim; ++i) {
                                const real_t cf2 = gr + B2 + sqrt(cf1*cf1 + 4.0*gr*Bt2[i]);
                                const real_t cf = sqrt(0.5 * cf2 / q[IR]);
                      
                                const real_t cmax = fmax(std::abs(V[i] - cf), std::abs(V[i] + cf));
                                inv_dt_hyp_loc = fmax(inv_dt_hyp_loc, cmax/D[i]);
                              }
                              #endif

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

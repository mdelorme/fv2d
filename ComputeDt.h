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
    Kokkos::parallel_reduce("Computing DT",
                            full_params.range_dom,
                            KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
                              // Hydro time-step
                              State q = getStateFromArray(Q, i, j);
                              real_t cs = speedOfSound(q, params);
                              real_t inv_dt_hyp_loc = (cs + fabs(q[IU]))/params.dx + (cs + fabs(q[IV]))/params.dy;
                              if (mhd_run) // In this case, we compute the MHD time-step below.
                                inv_dt_hyp_loc = 0.0; 

                              real_t inv_dt_par_tc_loc = params.epsilon;
                              if (params.thermal_conductivity_active)
                                inv_dt_par_tc_loc = fmax(2.0*computeKappa(i, j, params) / (params.dx*params.dx),
                                                         2.0*computeKappa(i, j, params) / (params.dy*params.dy));
                              
                              real_t inv_dt_par_visc_loc = params.epsilon;
                              if (params.viscosity_active)
                                inv_dt_par_visc_loc = fmax(2.0*computeMu(i, j, params) / (params.dx*params.dx),
                                                           2.0*computeMu(i, j, params) / (params.dy*params.dy));
                              #ifdef MHD
                              // if (params.riemann_solver == FIVEWAVES) {
                                const real_t B2 = q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ];
                                // const real_t cs = speedOfSound(q, params);
                                const real_t cs2 = cs * cs;
                                const real_t ca2 = B2 / q[IR];
                                const real_t cap2x = q[IBX]*q[IBX]/q[IR];
                                const real_t cap2y = q[IBY]*q[IBY]/q[IR];
                                // const real_t cap2z = q[IBZ]*q[IBZ]/q[IR];

                                const real_t cf_x = sqrt(0.5*(cs2+ca2)+0.5*sqrt((cs2+ca2)*(cs2+ca2)-4.0*cs2*cap2x));
                                const real_t cf_y = sqrt(0.5*(cs2+ca2)+0.5*sqrt((cs2+ca2)*(cs2+ca2)-4.0*cs2*cap2y));
                                // const real_t c_jz = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2z));
                                inv_dt_hyp_loc = (cf_x + Kokkos::abs(q[IU])) / params.dx + (cf_y + Kokkos::abs(q[IV])) / params.dy;
                              // }
                              // else {
                              // const real_t Bx = q[IBX];
                              // const real_t By = q[IBY];
                              // const real_t Bz = q[IBZ];
                              // const real_t gr = cs*cs*q[IR];
                              // const real_t Bt2 [] = {By*By+Bz*Bz,
                              //                        Bx*Bx+Bz*Bz,
                              //                        Bx*Bx+By*By};
                              // const real_t B2 = Bx*Bx + By*By + Bz*Bz;
                              // const real_t cf1 = gr-B2;
                              // const real_t V [] = {q[IU], q[IV], q[IW]};
                              // const real_t dl [] = {params.dx, params.dy};
                              
                              // const int ndim = 2;
                              // for (int i=0; i < ndim; ++i) {
                              //   const real_t cf2 = gr + B2 + sqrt(cf1*cf1 + 4.0*gr*Bt2[i]);
                              //   const real_t cf = sqrt(0.5 * cf2 / q[IR]);
                      
                              //   const real_t lambdaMax = fmax(Kokkos::abs(V[i] - cf), Kokkos::abs(V[i] + cf));
                              //   inv_dt_hyp_loc = fmax(inv_dt_hyp_loc, lambdaMax/dl[i]);
                              //   }
                              // }
                              #endif

                              inv_dt_hyp      = fmax(inv_dt_hyp, inv_dt_hyp_loc);
                              inv_dt_par_tc   = fmax(inv_dt_par_tc, inv_dt_par_tc_loc);
                              inv_dt_par_visc = fmax(inv_dt_par_visc, inv_dt_par_visc_loc);
                            }, Kokkos::Max<real_t>(inv_dt_hyp), 
                               Kokkos::Max<real_t>(inv_dt_par_tc), 
                               Kokkos::Max<real_t>(inv_dt_par_visc));

    if (diag) {
      std::cout << "Computing dts at (t=" << t << ") : dt_hyp=" << full_params.device_params.CFL/inv_dt_hyp;
      if(params.thermal_conductivity_active)
        std::cout << "; dt_TC="   << full_params.device_params.CFL/inv_dt_par_tc;
      if(params.viscosity_active)
        std::cout << "; dt_visc=" << full_params.device_params.CFL/inv_dt_par_visc;
      std::cout << std::endl; 
    }
    
    return full_params.device_params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

}

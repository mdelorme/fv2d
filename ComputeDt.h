#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d
{

class ComputeDtFunctor
{
public:
  Params full_params;

  ComputeDtFunctor(const Params &full_params) : full_params(full_params) {};
  ~ComputeDtFunctor() = default;

  real_t computeDt(Array Q, real_t max_dt, real_t t, bool diag) const
  {
    auto params = full_params.device_params;

    real_t inv_dt_hyp      = 0.0;
    real_t inv_dt_par_tc   = 0.0;
    real_t inv_dt_par_visc = 0.0;
    Kokkos::parallel_reduce(
        "Computing DT",
        full_params.range_dom,
        KOKKOS_LAMBDA(int i, int j, real_t &inv_dt_hyp, real_t &inv_dt_par_tc, real_t &inv_dt_par_visc) {
          // Traditional Hydro time-step
          State q               = getStateFromArray(Q, i, j);
          real_t cs             = speedOfSound(q, params);
          real_t inv_dt_hyp_loc = (cs + fabs(q[IU])) / params.dx + (cs + fabs(q[IV])) / params.dy;
#ifdef MHD
          if (mhd_run) // In this case, we compute the MHD time-step below.
          {
            const real_t B2    = q[IBX] * q[IBX] + q[IBY] * q[IBY] + q[IBZ] * q[IBZ];
            const real_t cs2   = cs * cs;
            const real_t ca2   = B2 / q[IR];
            const real_t cap2x = q[IBX] * q[IBX] / q[IR];
            const real_t cap2y = q[IBY] * q[IBY] / q[IR];

            const real_t cf_x = sqrt(0.5 * (cs2 + ca2) + 0.5 * sqrt((cs2 + ca2) * (cs2 + ca2) - 4.0 * cs2 * cap2x));
            const real_t cf_y = sqrt(0.5 * (cs2 + ca2) + 0.5 * sqrt((cs2 + ca2) * (cs2 + ca2) - 4.0 * cs2 * cap2y));
            // const real_t c_jz = sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2z));
            inv_dt_hyp_loc = (cf_x + Kokkos::abs(q[IU])) / params.dx + (cf_y + Kokkos::abs(q[IV])) / params.dy;
          }
#endif // MHD
          real_t inv_dt_par_tc_loc = params.epsilon;
          if (params.thermal_conductivity_active)
            inv_dt_par_tc_loc = fmax(2.0 * computeKappa(i, j, params) / (params.dx * params.dx),
                                     2.0 * computeKappa(i, j, params) / (params.dy * params.dy));

          real_t inv_dt_par_visc_loc = params.epsilon;
          if (params.viscosity_active)
            inv_dt_par_visc_loc = fmax(2.0 * computeMu(i, j, params) / (params.dx * params.dx),
                                       2.0 * computeMu(i, j, params) / (params.dy * params.dy));

          inv_dt_hyp      = fmax(inv_dt_hyp, inv_dt_hyp_loc);
          inv_dt_par_tc   = fmax(inv_dt_par_tc, inv_dt_par_tc_loc);
          inv_dt_par_visc = fmax(inv_dt_par_visc, inv_dt_par_visc_loc);
        },
        Kokkos::Max<real_t>(inv_dt_hyp),
        Kokkos::Max<real_t>(inv_dt_par_tc),
        Kokkos::Max<real_t>(inv_dt_par_visc));

    if (diag)
    {
      std::cout << "Computing dts at (t=" << t << ") : dt_hyp=" << 1.0 / inv_dt_hyp;
      if (params.thermal_conductivity_active)
        std::cout << "; dt_TC=" << 1.0 / inv_dt_par_tc;
      if (params.viscosity_active)
        std::cout << "; dt_visc=" << 1.0 / inv_dt_par_visc;
      std::cout << std::endl;
    }

    return full_params.device_params.CFL / std::max({inv_dt_hyp, inv_dt_par_tc, inv_dt_par_visc});
  }
};

} // namespace fv2d

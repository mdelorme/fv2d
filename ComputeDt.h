#pragma once

#include "SimInfo.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv1d {

real_t compute_dt(Array &Q, real_t max_dt, real_t t, bool diag) {

  real_t inv_dt_hyp  = 0.0;
  real_t inv_dt_tc   = 0.0;
  real_t inv_dt_visc = 0.0;

  #pragma omp parallel reduction(max:inv_dt_hyp) reduction(max:inv_dt_tc) reduction(max:inv_dt_visc)
  for (int j=jbeg; j<jend; ++j) {
    for (int i=ibeg; i<iend; ++i) {
      Pos pos = get_pos(i, j);

      // Hyperbolic CFL
      real_t cs = speed_of_sound(Q[j][i]);
      inv_dt_hyp = std::max(inv_dt_hyp, (cs + std::fabs(Q[j][i][IU]))/dx);
      inv_dt_hyp = std::max(inv_dt_hyp, (cs + std::fabs(Q[j][i][IV]))/dy);
    
      // Parabolic
      if (thermal_conductivity_active) {
        inv_dt_tc = std::max(inv_dt_tc, 2.0 * compute_kappa(pos[IX], pos[IY]) / (dx*dx));
        inv_dt_tc = std::max(inv_dt_tc, 2.0 * compute_kappa(pos[IX], pos[IY]) / (dy*dy));
      }
      
      if (viscosity_active) {
        inv_dt_visc = std::max(inv_dt_visc, 2.0 * compute_mu(pos[IX], pos[IY]) / (dx*dx));
        inv_dt_visc = std::max(inv_dt_visc, 2.0 * compute_mu(pos[IY], pos[IY]) / (dy*dy));
      }
    }
  }

  if (diag) {
    std::cout << "Computing dts at (t=" << t << ") : dt_hyp=" << 1.0/inv_dt_hyp 
              << "; dt_TC="   << 1.0/inv_dt_tc 
              << "; dt_visc=" << 1.0/inv_dt_visc << std::endl; 
  }

  real_t dt = CFL / (std::max({inv_dt_hyp, inv_dt_tc, inv_dt_visc}));

  return std::min(dt, max_dt);
}

}
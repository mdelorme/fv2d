#pragma once

#include "SimInfo.h"

namespace fv1d {

real_t compute_kappa(real_t x, real_t y) {
  real_t res;
  switch (thermal_conductivity_mode) {
    case TCM_B02: 
    {
      real_t tr = (tanh((y-b02_ymid)/b02_thickness) + 1.0) * 0.5;
      res = kappa * (b02_kappa1 * (1.0-tr) + b02_kappa2 * tr);
      break;
    }
    default:
      res = kappa;
  }

  return res;
}

void apply_thermal_conduction(Array &Q, Array &Unew, real_t dt) {
  real_t ftop, fbot;
  
  #pragma omp parallel for
  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);
      real_t x = pos[IX];
      real_t y = pos[IY];

      real_t kappaL = 0.5 * (compute_kappa(x, y) + compute_kappa(x-dx, y));
      real_t kappaR = 0.5 * (compute_kappa(x, y) + compute_kappa(x+dx, y));
      real_t kappaU = 0.5 * (compute_kappa(x, y) + compute_kappa(x, y-dy));
      real_t kappaD = 0.5 * (compute_kappa(x, y) + compute_kappa(x, y+dy));

      // Ideal EOS with R = 1 assumed. T = P/rho
      real_t TC = Q[j][i][IP]   / Q[j][i][IR];
      real_t TL = Q[j][i-1][IP] / Q[j][i-1][IR];
      real_t TR = Q[j][i+1][IP] / Q[j][i+1][IR];
      real_t TU = Q[j-1][i][IP] / Q[j-1][i][IR];
      real_t TD = Q[j+1][i][IP] / Q[j+1][i][IR];

      // Computing thermal flux
      real_t FL = kappaL * (TC - TL) / dx;
      real_t FR = kappaR * (TR - TC) / dx;
      real_t FU = kappaU * (TC - TU) / dy;
      real_t FD = kappaD * (TD - TC) / dy;

      /** 
       * Boundaries treatment
       * IMPORTANT NOTE :
       * To be accurate, in the case of fixed temperature, since the temperature is taken at the interface
       * the value of kappa should either be averaged between the cell-centered value and the interface
       * or be evaluated at x=0.25dx / x=xmax-0.25dx
       */
      if (j==jbeg && bctc_ymin != BCTC_NONE) {
        switch (bctc_ymin) {
          case BCTC_FIXED_TEMPERATURE: FL = kappaL * 2.0 * (TC-bctc_ymin_value) / dy; break;
          case BCTC_FIXED_GRADIENT:    FL = kappaL * bctc_ymin_value; break;
          default: break;
        }
        ftop = FL;
      }

      if (j==jend-1 && bctc_ymax != BCTC_NONE) {
        switch (bctc_ymax) {
          case BCTC_FIXED_TEMPERATURE: FR = kappaR * 2.0 * (bctc_ymax_value-TC) / dy; break;
          case BCTC_FIXED_GRADIENT:    FR = kappaR * bctc_ymax_value; break;       
          default: break;
        }
        fbot = FR;
      }

      // And updating using a Godunov-like scheme
      Unew[j][i][IE] += dt/dx * (FR - FL) + dt/dy * (FD - FU);
    }
  }

  //std::cout << "Fluxes " << ftop << " " << fbot << std::endl;
}

}
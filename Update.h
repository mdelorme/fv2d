#pragma once 

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv1d {

void compute_slopes(const Array &Q, Array &slopesX, Array &slopesY) {
  for (int j=1; j < Nty-1; ++j) {
    for (int i=1; i < Ntx-1; ++i) {
      for (int ivar=0; ivar < Nfields; ++ivar) {
        real_t dL = Q[j][i][ivar]   - Q[j][i-1][ivar];
        real_t dR = Q[j][i+1][ivar] - Q[j][i][ivar];
        real_t dU = Q[j][i][ivar]   - Q[j-1][i][ivar];
        real_t dD = Q[j+1][i][ivar] - Q[j][i][ivar];


        auto minmod = [](real_t dL, real_t dR) -> real_t {
          if (dL*dR < 0.0)
            return 0.0;
          else if (fabs(dL) < fabs(dR))
            return dL;
          else
            return dR;
        };

        slopesX[j][i][ivar] = minmod(dL, dR);
        slopesY[j][i][ivar] = minmod(dU, dD);
      }
    }
  }
}

State reconstruct(State &q, State &slope, real_t sign, IDir dir) {
  State res;
  switch (reconstruction) {
    case PLM: res = q + slope * sign * 0.5; break; // Piecewise Linear
    case PCM_WB: // Piecewise constant + Well-balancing
      res[IR] = q[IR];
      res[IU] = q[IU];
      res[IV] = q[IV];
      res[IP] = q[IP] + sign * q[IR] * g * dy * 0.5;
      break;
    default:  res = q; // Piecewise Constant
  }

  return swap_component(res, dir);
}

void compute_fluxes_and_update(Array &Q, Array &slopesX, Array &slopesY, Array &Unew, real_t dt) {
  auto update_along_dir = [&](int i, int j, IDir dir) {
    auto& slopes = (dir == IX ? slopesX : slopesY);
    int dxm = (dir == IX ? -1 : 0);
    int dxp = (dir == IX ?  1 : 0);
    int dym = (dir == IY ? -1 : 0);
    int dyp = (dir == IY ?  1 : 0);

    State qCL = reconstruct(Q[j][i],   slopes[j][i], -1.0, dir);
    State qCR = reconstruct(Q[j][i],   slopes[j][i],  1.0, dir);
    State qL  = reconstruct(Q[j+dym][i+dxm], slopes[j+dym][i+dxm],  1.0, dir);
    State qR  = reconstruct(Q[j+dyp][i+dxp], slopes[j+dyp][i+dxp], -1.0, dir);

    auto riemann = [&](State qL, State qR, State &flux, real_t &pout) {
      switch (riemann_solver) {
        case HLL: hll(qL, qR, flux, pout); break;
        default: hllc(qL, qR, flux, pout); break;
      }
    };

    // Calculating flux left and right of the cell
    State fluxL, fluxR;
    real_t poutL, poutR;

    riemann(qL, qCL, fluxL, poutL);
    riemann(qCR, qR, fluxR, poutR);

    fluxL = swap_component(fluxL, dir);
    fluxR = swap_component(fluxR, dir);

    // Remove mechanical flux in a well-balanced fashion
    if (well_balanced_flux_at_y_bc && (j==jbeg || j==jend-1) && dir == IY) {
      if (j==jbeg)
        fluxL = State{0.0, 0.0, poutR - Q[j][i][IR]*g*dy, 0.0};
      else 
        fluxR = State{0.0, 0.0, poutL + Q[j][i][IR]*g*dy, 0.0};
    }

    Unew[j][i] += dt/(dir == IX ? dx : dy)*(fluxL - fluxR);
  
    if (dir == IY && gravity) {
      Unew[j][i][IV] += dt * Q[j][i][IR] * g;
      Unew[j][i][IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * g;
    }
  };

  #pragma omp parallel for
  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      update_along_dir(i, j, IX);
      update_along_dir(i, j, IY);

      Unew[j][i][IR] = std::max(1.0e-6, Unew[j][i][IR]);
    }
  }

}


void update(Array &Q, Array &Unew, real_t dt) {
  // First filling up boundaries for ghosts terms
  fill_boundaries(Q, dt);

  // Hyperbolic update
  Array slopesX, slopesY;
  allocate_array(slopesX);
  allocate_array(slopesY);

  compute_slopes(Q, slopesX, slopesY);
  compute_fluxes_and_update(Q, slopesX, slopesY, Unew, dt);

  // Splitted terms
  if (thermal_conductivity_active)
    apply_thermal_conduction(Q, Unew, dt);
  if (viscosity_active)
    apply_viscosity(Q, Unew, dt);
}

}
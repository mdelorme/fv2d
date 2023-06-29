#pragma once

#include "SimInfo.h"

namespace fv1d {

real_t compute_mu(real_t x, real_t y) {
  switch (viscosity_mode) {
    default: return mu; break;
  }
}

void apply_viscosity(Array &Q, Array &Unew, real_t dt) {
  #pragma omp parallel for
  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);
      real_t x = pos[IX];
      real_t y = pos[IY];

      State stencil[3][3];

      // Computing viscous fluxes
      constexpr real_t four_thirds = 4.0/3.0;
      constexpr real_t two_thirds  = 2.0/3.0;

      auto fill_stencil = [&](int i, int j) -> void {
        for (int di=-1; di < 2; ++di)
          for (int dj=-1; dj < 2; ++dj)
            stencil[dj+1][di+1] = Q[j+dj][i+di];
      };

      auto compute_viscous_flux = [&](IDir dir) {
        State flux {0.0, 0.0, 0.0, 0.0};

        // Here X is the normal component and Y the tangential
        const real_t one_over_dx = 1.0/dx;
        const real_t one_over_dy = 1.0/dy;

        real_t mu = compute_mu(x, y);
        fill_stencil(i, j);

        for (int side=1; side < 3; ++side) {
          real_t sign = (side == 1 ? -1.0 : 1.0);
          
          if (dir == IX) {
            State qi = 0.5 * (stencil[1][side] + stencil[1][side-1]);

            real_t dudx = one_over_dx * (stencil[1][side][IU] - stencil[1][side-1][IU]);
            real_t dvdx = one_over_dx * (stencil[1][side][IV] - stencil[1][side-1][IV]);
            real_t dudy = 0.25 * one_over_dy * (stencil[2][side][IU]   - stencil[0][side][IU]
                                              + stencil[2][side-1][IU] - stencil[0][side-1][IU]);
            real_t dvdy = 0.25 * one_over_dy * (stencil[2][side][IV]   - stencil[0][side][IV]
                                              + stencil[2][side-1][IV] - stencil[0][side-1][IV]);

            const real_t tau_xx = four_thirds * dudx - two_thirds * dvdy;
            const real_t tau_xy = dvdx + dudy;

            flux[IU] += sign * mu * tau_xx;
            flux[IV] += sign * mu * tau_xy;
            flux[IE] += sign * mu * (tau_xx*qi[IU] + tau_xy*qi[IV]);
          }
          else if (dir == IY) {
            State qi = 0.5 * (stencil[side][1] + stencil[side-1][1]);

            real_t dudy = one_over_dy * (stencil[side][1][IU] - stencil[side-1][1][IU]);
            real_t dvdy = one_over_dy * (stencil[side][1][IV] - stencil[side-1][1][IV]);
            real_t dudx = 0.25 * one_over_dx * (stencil[side][2][IU]   - stencil[side][0][IU]
                                             +  stencil[side-1][2][IU] - stencil[side-1][0][IU]);
            real_t dvdx = 0.25 * one_over_dx * (stencil[side][2][IV]   - stencil[side][0][IV]
                                             +  stencil[side-1][2][IV] - stencil[side-1][0][IV]);
                                            
            const real_t tau_yy = four_thirds * dvdy - two_thirds * dudx;
            const real_t tau_xy = dvdx + dudy;


            flux[IU] += sign * mu * tau_xy;
            flux[IV] += sign * mu * tau_yy;
            flux[IE] += sign * mu * (tau_xy*qi[IU] + tau_yy*qi[IV]);
          } 
        }

        return flux;
      };

      State vf_x = compute_viscous_flux(IX);
      State vf_y = compute_viscous_flux(IY);

      Unew[j][i] += dt * (vf_x + vf_y);

    }
  }
}

}
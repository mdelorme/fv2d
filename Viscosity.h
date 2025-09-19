#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t computeMu(int i, int j, const DeviceParams &params) {
  real_t res;
  switch (params.viscosity_mode) {
    // default:
    //   {
        // const real_t y1 = params.iso3_dy0;
        // const real_t y2 = y1 + params.iso3_dy1;
        // const real_t y = getPos(params, i, j)[IY];
        // const real_t th = 0.05; // Thickness of transition layer
        // const real_t tr1 = (tanh((y-y1)/th) + 1.0) * 0.5;
        // const real_t tr2 = (tanh((y2-y)/th) + 1.0) * 0.5;
        // const real_t tr = tr1;
        // res = params.mu * (1.0 * (1.0-tr) + 10.0 * tr);
      // break;
      // }
    default: res = params.mu; break;
  }
  return res;
}

class ViscosityFunctor {
public:
  Params full_params;

  ViscosityFunctor(const Params &full_params) 
    : full_params(full_params) {};
  ~ViscosityFunctor() = default;

  void applyViscosity(Array Q, Array Unew, real_t dt, int ite) {
    auto params = full_params.device_params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    real_t total_viscous_contrib = 0.0;

    Kokkos::parallel_reduce(
      "Viscosity",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, real_t &viscous_contrib) {
        Pos pos = getPos(params, i, j);
        real_t x = pos[IX];
        real_t y = pos[IY];

        State stencil[3][3];

        // Computing viscous fluxes
        constexpr real_t four_thirds = 4.0/3.0;
        constexpr real_t two_thirds  = 2.0/3.0;

        auto fillStencil = [&](int i, int j) -> void {
          for (int di=-1; di < 2; ++di)
            for (int dj=-1; dj < 2; ++dj)
              stencil[dj+1][di+1] = getStateFromArray(Q, i+di, j+dj);
        };

        auto computeViscousFlux = [&](IDir dir) {
          State flux = zero_state();
          // Here X is the normal component and Y the tangential
          const real_t one_over_dx = 1.0/dx;
          const real_t one_over_dy = 1.0/dy;

          real_t mu = computeMu(i, j, params);
          fillStencil(i, j);

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

        State vf_x = computeViscousFlux(IX);
        State vf_y = computeViscousFlux(IY);

        State un_loc = getStateFromArray(Unew, i, j);
        un_loc[IU] += dt * (vf_x[IU] + vf_y[IU]);
        un_loc[IV] += dt * (vf_x[IV] + vf_y[IV]);
        un_loc[IE] += dt * (vf_x[IE] + vf_y[IE]);

        viscous_contrib += dt * (vf_x[IE] + vf_y[IE]);
        setStateInArray(Unew, i, j, un_loc);

      }, Kokkos::Sum<real_t>(total_viscous_contrib));
  }
};


}

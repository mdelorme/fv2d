#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t computeKappa(int i, int j, const DeviceParams &params) {
  real_t res;
  switch (params.thermal_conductivity_mode) {
    case TCM_B02:
    {
      const real_t y = getPos(params, i, j)[IY];
      const real_t tr = (tanh((y-params.b02_ymid)/params.b02_thickness) + 1.0) * 0.5;
      res = params.kappa * (params.b02_kappa1 * (1.0-tr) + params.b02_kappa2 * tr);
      break;
    }
    case TCM_C2020_TRI:
    {
      const real_t y = getPos(params, i, j)[IY];
      const real_t th = 0.05; // Thickness of transition layer
      const real_t tr1 = (tanh((y-params.tri_y1)/th) + 1.0) * 0.5;
      const real_t tr2 = (tanh((params.tri_y2-y)/th) + 1.0) * 0.5;
      const real_t tr = tr1*tr2;
      res = params.kappa * (params.tri_k2 * (1.0-tr) + params.tri_k1 * tr);
      break;
    }
    case TCM_ISO3:
    {
      const real_t y1 = params.iso3_dy0;
      const real_t y2 = y1 + params.iso3_dy1;
      const real_t y = getPos(params, i, j)[IY];
      const real_t th = 0.05; // Thickness of transition layer
      const real_t tr1 = (tanh((y-y1)/th) + 1.0) * 0.5;
      const real_t tr2 = (tanh((y2-y)/th) + 1.0) * 0.5;
      const real_t tr = tr1*tr2;
      res = params.kappa * (params.iso3_k2 * (1.0-tr) + params.iso3_k1 * tr);
      break;
      // const real_t tr = (tanh((y-y2)/th) + 1.0) * 0.5;
      // res = params.kappa * (params.iso3_k1 * (1.0-tr) + params.iso3_k2 * tr);
      // break;
    }
    /*case TCM_C20_STABLE:
    {
      const real_t y = getPos(params, i, j)[IY];
      const real_t H = params.c20_H;
      const real_t tr1 = (tanh(y-H)/params.c20_tr_thick) + 1.0) * 0.5;
      const real_t tr2 = (tanh(params.ymax-H)/params.c20_tr_thick) + 1.0) * 0.5;
      res = params.kappa * (params.c20_K1*(1.0-tr1*tr2) + params.c20_K2*(tr1*tr2));
    }*/
    default:
      res = params.kappa;
  }

  return res;
}

class ThermalConductionFunctor {
public:
  Params full_params;

  ThermalConductionFunctor(const Params &full_params) 
    : full_params(full_params) {};
  ~ThermalConductionFunctor() = default;

  void applyThermalConduction(Array Q, Array Unew, real_t dt, int ite) {
    auto params = full_params.device_params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;

    real_t total_thermal_contrib = 0.0;

    Kokkos::parallel_reduce(
      "Thermal conduction", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, real_t &thermal_contrib) {
        Pos pos = getPos(params, i, j);
        real_t x = pos[IX];
        real_t y = pos[IY];

        real_t kappaL = 0.5 * (computeKappa(i, j, params) + computeKappa(i-1, j, params));
        real_t kappaR = 0.5 * (computeKappa(i, j, params) + computeKappa(i+1, j, params));
        real_t kappaU = 0.5 * (computeKappa(i, j, params) + computeKappa(i, j-1, params));
        real_t kappaD = 0.5 * (computeKappa(i, j, params) + computeKappa(i, j+1, params));

        // Ideal EOS with R = 1 assumed. T = P/rho
        real_t TC = Q(j, i, IP)   / Q(j, i,   IR);
        real_t TL = Q(j, i-1, IP) / Q(j, i-1, IR);
        real_t TR = Q(j, i+1, IP) / Q(j, i+1, IR);
        real_t TU = Q(j-1, i, IP) / Q(j-1, i, IR);
        real_t TD = Q(j+1, i, IP) / Q(j+1, i, IR);

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
        if (j==params.jbeg && params.bctc_ymin != BCTC_NONE) {
          switch (params.bctc_ymin) {
            case BCTC_FIXED_TEMPERATURE: FU = kappaU * 2.0 * (TC-params.bctc_ymin_value) / dy; break;
            case BCTC_FIXED_GRADIENT:    FU = kappaU * params.bctc_ymin_value; break;
            case BCTC_NO_CONDUCTION:     FU = FD; break;
            case BCTC_NO_FLUX:           FU = 0.0; break;
            default: break;
          }
        }

        if (j==params.jend-1 && params.bctc_ymax != BCTC_NONE) {
          switch (params.bctc_ymax) {
            case BCTC_FIXED_TEMPERATURE: FD = kappaD * 2.0 * (params.bctc_ymax_value-TC) / dy; break;
            case BCTC_FIXED_GRADIENT:    FD = kappaD * params.bctc_ymax_value; break;
            case BCTC_NO_CONDUCTION:     FD = FU; break;
            case BCTC_NO_FLUX:           FD = 0.0; break;
            default: break;
          }
        }

        // And updating using a Godunov-like scheme
        Unew(j, i, IE) += dt/dx * (FR - FL) + dt/dy * (FD - FU);
        thermal_contrib += dt/dx * (FR - FL) + dt/dy * (FD - FU);
      }, Kokkos::Sum<real_t>(total_thermal_contrib));

    if (params.log_energy_contributions && ite % params.log_energy_frequency == 0)
      std::cout << "Total thermal contribution to energy : " << total_thermal_contrib << std::endl;
  }
};

}

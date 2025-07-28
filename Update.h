#pragma once

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Heating.h"
#include "Viscosity.h"
#include "Sources.h"
#include "Gravity.h"

namespace fv2d {

namespace {
  KOKKOS_INLINE_FUNCTION
  State reconstruct(Array Q, Array slopes, int i, int j, real_t sign, IDir dir, const DeviceParams &params) {
    State q     = getStateFromArray(Q, i, j);
    State slope = getStateFromArray(slopes, i, j);
    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * sign * 0.5; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = q[IP] + sign * q[IR] * getGravity(i, j, dir, params) * params.dy * 0.5;
        #ifdef MHD
        res[IW] = q[IW];
        res[IBX] = q[IBX];
        res[IBY] = q[IBY];
        res[IBZ] = q[IBZ];
        res[IPSI] = q[IPSI];
        #endif // MHD
      default:  res = q; // Piecewise Constant
    }

    return swap_component(res, dir);
  }
}

class UpdateFunctor {
public:
  Params full_params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;
  SourcesFunctor sources_functor;
  HeatingFunctor heat_functor;

  Array slopesX, slopesY;

  UpdateFunctor(const Params &full_params)
    : full_params(full_params), bc_manager(full_params),
      tc_functor(full_params), visc_functor(full_params),
      heat_functor(full_params), sources_functor(full_params) {
      auto device_params = full_params.device_params;
      slopesX = Array("SlopesX", device_params.Nty, device_params.Ntx, Nfields);
      slopesY = Array("SlopesY", device_params.Nty, device_params.Ntx, Nfields);

      // if (mhd_run && (device_params.riemann_solver==HLL || device_params.riemann_solver==HLLC)){
      //   throw std::runtime_error("HLL and HLLC are not supported for MHD runs.");
      // }
    };
  ~UpdateFunctor() = default;

  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    Kokkos::parallel_for(
      "Slopes",
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        for (int ivar=0; ivar < Nfields; ++ivar) {
          real_t dL = Q(j, i, ivar)   - Q(j, i-1, ivar);
          real_t dR = Q(j, i+1, ivar) - Q(j, i, ivar);
          real_t dU = Q(j, i, ivar)   - Q(j-1, i, ivar);
          real_t dD = Q(j+1, i, ivar) - Q(j, i, ivar);

          auto minmod = [](real_t dL, real_t dR) -> real_t {
            if (dL*dR < 0.0)
              return 0.0;
            else if (fabs(dL) < fabs(dR))
              return dL;
            else
              return dR;
          };

          slopesX(j, i, ivar) = minmod(dL, dR);
          slopesY(j, i, ivar) = minmod(dU, dD);
        }
      });

  }

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt, real_t GLM_ch1) const {
    auto params = full_params.device_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    // if (params.riemann_solver == IDEALGLM)
    //   ch_derigs = ComputeGlobalDivergenceSpeed(Q, full_params);
    
    Kokkos::parallel_for(
      "Update", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        // Lambda to update the cell along a direction
        const real_t ch_derigs = params.GLM_scale * GLM_ch1/dt;
        const real_t ch_dedner = 0.5 * params.CFL * fmin(params.dx, params.dy)/dt;
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : slopesY);
          int dxm = (dir == IX ? -1 : 0);
          int dxp = (dir == IX ?  1 : 0);
          int dym = (dir == IY ? -1 : 0);
          int dyp = (dir == IY ?  1 : 0);
          
          State qCL = reconstruct(Q, slopes, i, j, -1.0, dir, params);
          State qCR = reconstruct(Q, slopes, i, j,  1.0, dir, params);
          State qL  = reconstruct(Q, slopes, i+dxm, j+dym, 1.0, dir, params);
          State qR  = reconstruct(Q, slopes, i+dxp, j+dyp, -1.0, dir, params);
          
          // Calling the right Riemann solver
          auto riemann = [&](State qL, State qR, State &flux, real_t &pout) {
            #ifdef MHD
            real_t Bx_m, psi_m;
            if (params.div_cleaning == DEDNER) {
              Bx_m  = qL[IBX]  + 0.5 * (qR[IBX] - qL[IBX]) - 1/(2*ch_dedner) * (qR[IPSI] - qL[IPSI]);
              psi_m = qL[IPSI] + 0.5 * (qR[IPSI] - qL[IPSI]) - 0.5*ch_dedner * (qR[IBX] - qL[IBX]);
            } 
            else {
              Bx_m = qL[IBX] + 0.5 * (qR[IBX] - qL[IBX]);
              psi_m = 0.0;
            }
            
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params);   break;
              case HLLD: {
                hlld(qL, qR, flux, pout, Bx_m, params);
                break;
              }
              case FIVEWAVES: {
                FiveWaves(qL, qR, flux, pout, params);
                break;
              }
              case IDEALGLM: {
                IdealGLM(qL, qR, flux, pout, ch_derigs, params);
                break;
              }
              default: hlld(qL, qR, flux, pout, Bx_m, params);   break;
            }
            if (params.div_cleaning == DEDNER){
              flux[IBX] = psi_m;
              flux[IPSI] = ch_dedner*ch_dedner*Bx_m;
            }
            #else
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params);   break;
              default: hllc(qL, qR, flux, pout, params);   break;
            }
            #endif // MHD
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL);
          riemann(qCR, qR, fluxR, poutR);

          fluxL = swap_component(fluxL, dir);
          fluxR = swap_component(fluxR, dir);
          
          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_y_bc &&  (j==params.jbeg || j==params.jend-1) && dir == IY) {
            State q = getStateFromArray(Q, i, j);
            real_t g = getGravity(i, j, dir, params);
            if (j==params.jbeg){
              fluxL = zero_state();
              fluxL[IV] = poutR - Q(j, i, IR)*g*params.dy;
              #ifdef MHD
              fluxL[IV] -= 0.5 * q[IBY]*q[IBY];
              fluxL[IBX] = - q[IU] * q[IBY];
              fluxL[IBZ] = - q[IW] * q[IBY];
              #endif // MHD
            }
            else{
              fluxR = zero_state();
              fluxR[IV] = poutL + Q(j, i, IR)*g*params.dy;
              #ifdef MHD
              fluxR[IV] -= 0.5 * q[IBY]*q[IBY];
              fluxR[IBX] = - q[IU] * q[IBY];
              fluxR[IBZ] = - q[IW] * q[IBY];
              #endif
            }
          }
          

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(fluxL - fluxR)/(dir == IX ? params.dx : params.dy);

          if (params.gravity_mode != GRAV_NONE) {
            real_t g = getGravity(i, j, dir, params);
            un_loc[IV] += dt * Q(j, i, IR) * g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * g;
          }
          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        Unew(j, i, IR) = fmax(params.smallr, Unew(j, i, IR));
      });
  }
  

  void euler_step(Array Q, Array Unew, real_t dt, real_t GLM_ch1) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);
    // Hyperbolic update
    if (full_params.device_params.reconstruction == PLM)
      computeSlopes(Q);
    computeFluxesAndUpdate(Q, Unew, dt, GLM_ch1);
    // Splitted terms
    if (full_params.device_params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (full_params.device_params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
    if (full_params.device_params.heating_active)
      heat_functor.applyHeating(Q, Unew, dt);
    sources_functor.applySources(Q, Unew, dt, GLM_ch1);
    // auto params = full_params.device_params;
    // Kokkos::parallel_for(
    //     "Clean values", 
    //     full_params.range_dom,
    //     KOKKOS_LAMBDA(const int i, const int j) {
    //       auto uloc = getStateFromArray(Unew, i, j);
    //       auto qloc = consToPrim(uloc, params);
    //       qloc[IR] = Kokkos::max(qloc[IR], 1.0e-10);
    //       qloc[IP] = Kokkos::max(qloc[IP], 1.0e-10);
    //       uloc = primToCons(qloc, params);
    //       setStateInArray(Unew, i, j, uloc);
    //     });
  }


  void update(Array Q, Array Unew, real_t dt, real_t GLM_ch1) {
    if (full_params.time_stepping == TS_EULER)
      euler_step(Q, Unew, dt, GLM_ch1);
    else if (full_params.time_stepping == TS_RK2) {
      auto params = full_params.device_params;
      Array U0    = Array("U0", params.Nty, params.Ntx, Nfields);
      Array Ustar = Array("Ustar", params.Nty, params.Ntx, Nfields);

      // Step 1
      Kokkos::deep_copy(U0, Unew);
      Kokkos::deep_copy(Ustar, Unew);
      euler_step(Q, Ustar, dt, GLM_ch1);
      // Step 2
      Kokkos::deep_copy(Unew, Ustar);
      consToPrim(Ustar, Q, full_params);
      euler_step(Q, Unew, dt, GLM_ch1);
      // SSP-RK2
      Kokkos::parallel_for(
        "RK2 Correct", 
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          for (int ivar=0; ivar < Nfields; ++ivar)
            Unew(j, i, ivar) = 0.5 * (U0(j, i, ivar) + Unew(j, i, ivar));
        });
    }
  }
};

}

#pragma once

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

namespace {
  KOKKOS_INLINE_FUNCTION
  State reconstruct(Array Q, Array slopes, int i, int j, real_t sign, IDir dir, const Params &params) {
    State q     = getStateFromArray(Q, i, j);
    State slope = getStateFromArray(slopes, i, j);
    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * sign * 0.5; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = (dir == IX ? q[IP] : q[IP] + sign * q[IR] * params.g * params.dy * 0.5);
        // TODO Lucas : Ajouter les éléments mhd si le run est mhd (en constexpr)
        break;
      default:  res = q; // Piecewise Constant
    }

    return swap_component(res, dir);
  }
}

class UpdateFunctor {
public:
  Params params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;

  Array slopesX, slopesY;

  UpdateFunctor(const Params &params)
    : params(params), bc_manager(params),
      tc_functor(params), visc_functor(params) {
      slopesX = Array("SlopesX", params.Nty, params.Ntx, Nfields);
      slopesY = Array("SlopesY", params.Nty, params.Ntx, Nfields);

      if (mhd_run && (params.riemann_solver==HLL || params.riemann_solver==HLLC)){
        throw std::runtime_error("HLL and HLLC are not supported for MHD runs.");
      }
    };
  ~UpdateFunctor() = default;

  KOKKOS_INLINE_FUNCTION
  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto params  = this->params;

    Kokkos::parallel_for(
      "Slopes",
      params.range_slopes,
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
  KOKKOS_INLINE_FUNCTION
  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    Kokkos::parallel_for(
      "Update", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        // Lambda to update the cell along a direction
        #ifdef MHD
        const real_t ch = 0.5 * params.CFL * std::min(params.dx, params.dy)/dt;
        const real_t cr = 0.1; // TODO: à mettre dans les paramètres
        const real_t cp = std::sqrt(cr*ch);
        const real_t parabolic = std::exp(-dt*ch*ch/(cp*cp));
        #endif

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
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params);   break;
              case HLLD: {
                // We first compute Bx_m and phi_m for the GLMMHD solver
                const real_t Bx_m = qL[IBX] + 0.5 * (qR[IBX] - qL[IBX]) - 1/(2*ch) * (qR[IPHI] - qL[IPHI]);
                hlld(qL, qR, flux, pout, Bx_m, params);
                const real_t psi_m = qL[IPHI] + 0.5 * (qR[IPHI] - qL[IPHI]) - 0.5*ch * (qR[IBX] - qL[IBX]);
                flux[IBX] = psi_m;
                flux[IPHI] = ch*ch*Bx_m;
                break;
              }
              default: hllc(qL, qR, flux, pout, params);   break;
            }
            #else
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params);   break;
              default: hllc(qL, qR, flux, pout, params);   break;
            }
            #endif
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL);
          riemann(qCR, qR, fluxR, poutR);

          fluxL = swap_component(fluxL, dir);
          fluxR = swap_component(fluxR, dir);

          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
            if (j==params.jbeg)
              #ifdef MHD
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
              #else
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0};
              #endif
            else
              #ifdef MHD
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
              #else
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0};
              #endif
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(fluxL - fluxR)/(dir == IX ? params.dx : params.dy);
          if (dir == IY && params.gravity) {
            un_loc[IV] += dt * Q(j, i, IR) * params.g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
          }
          setStateInArray(Unew, i, j, un_loc);
        };
        // #ifdef MHD
        // Q(j, i, IPHI) *= parabolic;
        // #endif
        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
        #ifdef MHD
        Unew(j, i, IPHI) *= parabolic;
        #endif
      });
  }

  void euler_step(Array Q, Array Unew, real_t dt) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hypperbolic udpate
    if (params.reconstruction == PLM)
      computeSlopes(Q);
    computeFluxesAndUpdate(Q, Unew, dt);

    // Splitted terms
    if (params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
  }

  void update(Array Q, Array Unew, real_t dt) {
    if (params.time_stepping == TS_EULER)
      euler_step(Q, Unew, dt);
    else if (params.time_stepping == TS_RK2) {
      Array U0    = Array("U0", params.Nty, params.Ntx, Nfields);
      Array Ustar = Array("Ustar", params.Nty, params.Ntx, Nfields);
      
      // Step 1
      Kokkos::deep_copy(U0, Unew);
      Kokkos::deep_copy(Ustar, Unew);
      euler_step(Q, Ustar, dt);
      
      // Step 2
      Kokkos::deep_copy(Unew, Ustar);
      consToPrim(Ustar, Q, params);
      euler_step(Q, Unew, dt);

      // SSP-RK2
      Kokkos::parallel_for(
        "RK2 Correct", 
        params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          for (int ivar=0; ivar < Nfields; ++ivar)
            Unew(j, i, ivar) = 0.5 * (U0(j, i, ivar) + Unew(j, i, ivar));
        });
    }
  }
};

}

#pragma once

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Heating.h"
#include "Viscosity.h"

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
        res[IP] = (dir == IX ? q[IP] : q[IP] + sign * q[IR] * params.g * params.dy * 0.5);
        break;
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
  HeatingFunctor heat_functor;

  Array slopesX, slopesY;

  UpdateFunctor(const Params &full_params)
    : full_params(full_params), bc_manager(full_params),
      tc_functor(full_params), visc_functor(full_params), heat_functor(full_params) {
      auto device_params = full_params.device_params;
      slopesX = Array("SlopesX", device_params.Nty, device_params.Ntx, Nfields);
      slopesY = Array("SlopesY", device_params.Nty, device_params.Ntx, Nfields);
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

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt, int ite) const {
    auto params = full_params.device_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    real_t total_hydro_contrib = 0.0;

    Kokkos::parallel_reduce(
      "Update", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, real_t &hydro_contrib) {
        // Lambda to update the cell along a direction
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
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params); break;
              default: hllc(qL, qR, flux, pout, params); break;
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
          if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
            if (j==params.jbeg)
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0};
            else
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0};
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          const real_t dh = (dir == IX ? params.dx : params.dy);
          un_loc += dt*(fluxL - fluxR)/dh;

          hydro_contrib += dt * (fluxL[IE]-fluxR[IE])/dh;

          if (dir == IY && params.gravity) {
            un_loc[IV] += dt * Q(j, i, IR) * params.g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
          }

          /*if (params.well_balanced_flux_at_y_bc && j==params.jbeg && dir == IY) {
            auto evol = dt*(fluxL-fluxR)/dh;
            evol[IV] += dt * Q(j, i, IR) * params.g;
            evol[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
            printf("At jbeg: FL = %lf %lf %lf %lf; FR = %lf %lf %lf %lf; gcontrib = %lf %lf; evol = %lf %lf %lf %lf\n",
                    fluxL[IR], fluxL[IU], fluxL[IV], fluxL[IE],
                    fluxR[IR], fluxR[IU], fluxR[IV], fluxR[IE],
                    dt * Q(j, i, IR) * params.g, dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g,
                    evol[IR], evol[IU], evol[IV], evol[IP]);
          }

          if (params.well_balanced_flux_at_y_bc && j==params.jbeg+1 && dir == IY) {
            auto evol = dt*(fluxL-fluxR)/dh;
            evol[IV] += dt * Q(j, i, IR) * params.g;
            evol[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
            printf("At jbeg+1: FL = %lf %lf %lf %lf; FR = %lf %lf %lf %lf; gcontrib = %lf %lf; evol = %lf %lf %lf %lf\n",
                    fluxL[IR], fluxL[IU], fluxL[IV], fluxL[IE],
                    fluxR[IR], fluxR[IU], fluxR[IV], fluxR[IE],
                    dt * Q(j, i, IR) * params.g, dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g,
                    evol[IR], evol[IU], evol[IV], evol[IP]);
          }*/
          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
      }, Kokkos::Sum<real_t>(total_hydro_contrib));

    if (full_params.log_energy_contributions && ite % full_params.log_energy_frequency == 0)
      std::cout << "Total hydro contribution to energy : " << total_hydro_contrib << std::endl;
  }

  void euler_step(Array Q, Array Unew, real_t dt, int ite) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hypperbolic udpate
    if (full_params.device_params.reconstruction == PLM)
      computeSlopes(Q);
    computeFluxesAndUpdate(Q, Unew, dt, ite);

    // Splitted terms
    if (full_params.device_params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt, ite);
    if (full_params.device_params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt, ite);
    if (full_params.device_params.heating_active)
      heat_functor.applyHeating(Q, Unew, dt, ite);
  }

  void update(Array Q, Array Unew, real_t dt, int ite) {
    if (full_params.time_stepping == TS_EULER)
      euler_step(Q, Unew, dt, ite);
    else if (full_params.time_stepping == TS_RK2) {
      auto params = full_params.device_params;
      Array U0    = Array("U0", params.Nty, params.Ntx, Nfields);
      Array Ustar = Array("Ustar", params.Nty, params.Ntx, Nfields);

      // Step 1
      Kokkos::deep_copy(U0, Unew);
      Kokkos::deep_copy(Ustar, Unew);
      euler_step(Q, Ustar, dt, ite);

      // Step 2
      Kokkos::deep_copy(Unew, Ustar);
      consToPrim(Ustar, Q, full_params);
      euler_step(Q, Unew, dt, ite);

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

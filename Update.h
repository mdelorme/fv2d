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
      case HANCOCK: 
      case PLM: res = q + slope * sign * 0.5; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = (dir == IX ? q[IP] : q[IP] + sign * q[IR] * params.g * params.dy * 0.5);
        break;
      default:  res = q; // Piecewise Constant
    }

    return res;
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
    };
  ~UpdateFunctor() = default;

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

  void computeHancockSources(const Array &Q, real_t dt) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto params  = this->params;

    const real_t dtdx = 0.5 * dt / params.dx;
    const real_t dtdy = 0.5 * dt / params.dy;
    const real_t gamma = params.gamma0;

    Kokkos::parallel_for(
      "MUSCL-Hancock",
      params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        auto [ r,   u,   v,   p ] = getStateFromArray(Q, i, j);
        auto [drx, dux, dvx, dpx] = getStateFromArray(slopesX, i, j);
        auto [dry, duy, dvy, dpy] = getStateFromArray(slopesY, i, j);
        
        Q(j, i, IR) = r - (u * drx + r * dux)         * dtdx - (v * dry + r * dvy)         * dtdy;
        Q(j, i, IU) = u - (u * dux + dpx / r)         * dtdx - (v * duy)                   * dtdy;
        Q(j, i, IV) = v - (u * dvx)                   * dtdx - (v * dvy + dpy / r)         * dtdy;
        Q(j, i, IP) = p - (gamma * p * dux + u * dpx) * dtdx - (gamma * p * dvy + v * dpy) * dtdy;
      });
  }

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    Kokkos::parallel_for(
      "Update", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
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
          auto riemann = [&](State qL, State qR, State &flux, real_t &pout, IDir dir) {
            qL = swap_component(qL, dir);
            qR = swap_component(qR, dir);
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params); break;
              default: hllc(qL, qR, flux, pout, params); break;
            }
            flux = swap_component(flux, dir);
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL, dir);
          riemann(qCR, qR, fluxR, poutR, dir);

          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
            if (j==params.jbeg)
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0};
            else 
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0};
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(fluxL - fluxR)/(dir == IX ? params.dx : params.dy);
        
          if (dir == IY && params.gravity) {
            un_loc[IV] += dt * Q(j, i, IR) * params.g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
          }

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
      });
  }

  void computeFluxesAndUpdate_CTU(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    Array Fhat[2] = { Array("Fhat_x", params.Nty, params.Ntx, Nfields),
                      Array("Fhat_y", params.Nty, params.Ntx, Nfields) };

    // predictor
    Kokkos::parallel_for(
      "Update CTU predictor", 
      params.range_fluxes,
      KOKKOS_LAMBDA(const int i, const int j) {

        // Calling the right Riemann solver
        auto riemann = [&](State qL, State qR, State &flux, real_t &pout, IDir dir) {
          qL = swap_component(qL, dir);
          qR = swap_component(qR, dir);
          switch (params.riemann_solver) {
            case HLL: hll(qL, qR, flux, pout, params); break;
            default: hllc(qL, qR, flux, pout, params); break;
          }
          flux = swap_component(flux, dir);
        };

        // Lambda to update the cell along a direction
        auto updatePredictor = [&](int i, int j, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : slopesY);
          int dxm = (dir == IX ? -1 : 0);
          int dym = (dir == IY ? -1 : 0);

          State qL = reconstruct(Q, slopes, i+dxm, j+dym,  1.0, dir, params);
          State qR = reconstruct(Q, slopes, i,     j,     -1.0, dir, params);

          // Calculating flux "hat" left
          State  flux;
          real_t pout;

          riemann(qL, qR, flux, pout, dir);
          setStateInArray(Fhat[dir], i, j, flux);
        };

        updatePredictor(i, j, IX);
        updatePredictor(i, j, IY);
      });

    // corrector
    Kokkos::parallel_for(
      "Update CTU corrector", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        State un_loc = getStateFromArray(Unew, i, j);

        // Calling the right Riemann solver
        auto riemann = [&](State qL, State qR, State &flux, real_t &pout, IDir dir) {
          qL = swap_component(qL, dir);
          qR = swap_component(qR, dir);
          switch (params.riemann_solver) {
            case HLL: hll(qL, qR, flux, pout, params); break;
            default: hllc(qL, qR, flux, pout, params); break;
          }
          flux = swap_component(flux, dir);
        };

        // Lambda to update the cell along a direction
        auto updateCorrector = [&](int i, int j, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : slopesY);
          int dxm = (dir == IX ? -1 : 0);
          int dxp = (dir == IX ?  1 : 0);
          int dym = (dir == IY ? -1 : 0);
          int dyp = (dir == IY ?  1 : 0);
          real_t dtddir  = dt/(dir == IX ? params.dx : params.dy);
          real_t dtdtdir = dt/(dir == IY ? params.dx : params.dy);
          IDir tdir = (dir == IX) ? IY : IX; // 2d case

          auto reconstructTransverse = [&](int ii, int jj, real_t sign) {
            State U = reconstruct(Q, slopes, ii, jj, sign, dir, params);
            U = primToCons(U, params);
            State FL = getStateFromArray(Fhat[tdir], ii,     jj);
            State FR = getStateFromArray(Fhat[tdir], ii+dyp, jj+dxp); // flux direction transverse
            U = U + 0.5 * dtdtdir * (FL - FR);
            return consToPrim(U, params);
          };

          State qCL = reconstructTransverse(i,     j,     -1.0);
          State qCR = reconstructTransverse(i,     j,      1.0);
          State qL  = reconstructTransverse(i+dxm, j+dym,  1.0);
          State qR  = reconstructTransverse(i+dxp, j+dyp, -1.0);
          
          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL, dir);
          riemann(qCR, qR, fluxR, poutR, dir);

          un_loc += dtddir * (fluxL - fluxR);
        };
        
        updateCorrector(i, j, IX);
        updateCorrector(i, j, IY);
        
        un_loc[IR] = fmax(1.0e-6, un_loc[IR]);
        setStateInArray(Unew, i, j, un_loc);
      });
  }

  void euler_step(Array Q, Array Unew, real_t dt) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hyperbolic udpate
    switch(params.reconstruction) { 
      case PLM:     computeSlopes(Q); break;
      case HANCOCK: computeSlopes(Q); computeHancockSources(Q, dt); break;
    }
    switch(params.timestepping_solver) { 
      case SOLVER_GOD: computeFluxesAndUpdate(Q, Unew, dt); break;
      case SOLVER_CTU: computeFluxesAndUpdate_CTU(Q, Unew, dt); break;
    }

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
      Array U0    = Array("U0",    params.Nty, params.Ntx, Nfields);
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
#pragma once 

#include "Geometry.h"
#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"
#include "Gravity.h"

namespace fv2d {

namespace {
  KOKKOS_INLINE_FUNCTION
  State reconstruct(State q, Array slopes, int i, int j, real_t length, IDir dir, const Params &params) {
    State slope = getStateFromArray(slopes, i, j);
    
    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * length; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = (dir == IX ? q[IP] : q[IP] + 2.0 * length * q[IR] * params.g * params.dy * 0.5);
        break;
      default:  res = q; // Piecewise Constant
    }

    return res;
    // return swap_component(res, dir);
  }

  KOKKOS_INLINE_FUNCTION
  State rotate(State q, Pos normale, IDir dir) {
    State res = q;
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t cos = normale[IX];
    const real_t sin = normale[IY];
  
    // printf("%s , %lf , %lf\n", (IX==dir) ? "IX" : "IY", cos, sin);
    res[IU] =  cos * u + sin * v;
    res[IV] = -sin * u + cos * v;
    return res;
  }
  
  KOKKOS_INLINE_FUNCTION
  State rotate_back(State q, Pos normale, IDir dir) {
    State res = q;
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t cos = normale[IX];
    const real_t sin = normale[IY];

    res[IU] =  cos * u - sin * v;
    res[IV] =  sin * u + cos * v;
    return res;
  }
}

class UpdateFunctor {
public:
  Params params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;
  GravityFunctor grav_functor;
  Geometry geometry;

  Array slopesX, slopesY;

  UpdateFunctor(const Params &params)
    : params(params), bc_manager(params),
      tc_functor(params), visc_functor(params), grav_functor(params),
      geometry(params) {
      
      slopesX = Array("SlopesX", params.Nty, params.Ntx, Nfields);
      slopesY = Array("SlopesY", params.Nty, params.Ntx, Nfields);
    };
  ~UpdateFunctor() = default;

  KOKKOS_INLINE_FUNCTION
  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto params  = this->params;
    auto geo = this->geometry;

    Kokkos::parallel_for(
      "Slopes",
      params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        for (int ivar=0; ivar < Nfields; ++ivar) {
          real_t dL = (Q(j, i, ivar)   - Q(j, i-1, ivar)) / geo.cellReconsLengthSlope(i,  j,  IX);
          real_t dR = (Q(j, i+1, ivar) - Q(j, i, ivar)  ) / geo.cellReconsLengthSlope(i+1,j,  IX);
          real_t dU = (Q(j, i, ivar)   - Q(j-1, i, ivar)) / geo.cellReconsLengthSlope(i,  j,  IY); 
          real_t dD = (Q(j+1, i, ivar) - Q(j, i, ivar)  ) / geo.cellReconsLengthSlope(i,  j+1,IY); 

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

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto geometry = this->geometry;

    Kokkos::parallel_for(
      "Update", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        const real_t cellArea = geometry.cellArea(i,j);
        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : slopesY);
          int dxm = (dir == IX ? -1 : 0);
          int dxp = (dir == IX ?  1 : 0);
          int dym = (dir == IY ? -1 : 0);
          int dyp = (dir == IY ?  1 : 0);

          ///////////////// Geometry things

          real_t lenL, lenR;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          Pos rotR = geometry.getRotationMatrix(i, j, dir, IRIGHT, lenR);

          auto [dLL, dLR] = geometry.cellReconsLength(i, j, dir);
          auto [dRL, dRR] = geometry.cellReconsLength(i+dxp, j+dyp, dir);
          
          State qCL, qCR, qL, qR;
          {
            State qC = getStateFromArray(Q, i, j);
                  qL = getStateFromArray(Q, i+dxm, j+dym);
                  qR = getStateFromArray(Q, i+dxp, j+dyp);

            #if 1
            // reconstruction before rotate
              qL  = reconstruct(qL, slopes, i+dxm, j+dym,  dLL, dir, params);
              qCL = reconstruct(qC, slopes, i,     j,     -dLR, dir, params);
              qCR = reconstruct(qC, slopes, i,     j,      dRL, dir, params);
              qR  = reconstruct(qR, slopes, i+dxp, j+dyp, -dRR, dir, params);

              qL  = rotate(qL , rotL, dir);
              qCL = rotate(qCL, rotL, dir);
              qCR = rotate(qCR, rotR, dir);
              qR  = rotate(qR , rotR, dir);
            #else
            // rotate before reconstruction
              qL  = rotate(qL , rotL, dir);
              qCL = rotate(qC,  rotL, dir);
              qCR = rotate(qC,  rotR, dir);
              qR  = rotate(qR , rotR, dir);

              qL  = reconstruct(qL, slopes, i+dxm, j+dym,  dLL, dir, params);
              qCL = reconstruct(qCL, slopes, i,    j,     -dLR, dir, params);
              qCR = reconstruct(qCR, slopes, i,    j,      dRL, dir, params);
              qR  = reconstruct(qR, slopes, i+dxp, j+dyp, -dRR, dir, params);
            #endif

          }

          /////////////// end Geometry

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

          // Remove mechanical flux in a well-balanced fashion
          // if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
          //   if (j==params.jbeg)
          //     fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0};
          //   else 
          //     fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0};
          // }

          // reflective fluxes
          if (dir == IX) {
            if (i==params.ibeg)
              fluxL = State{0.0, poutL, 0.0, 0.0};
            else if (i==params.iend-1)
              fluxR = State{0.0, poutR, 0.0, 0.0};
          }
          else {
            if (j==params.jbeg)
              fluxL = State{0.0, poutL, 0.0, 0.0};
            else if (j==params.jend-1)
              fluxR = State{0.0, poutR, 0.0, 0.0};
          }

          fluxL = rotate_back(fluxL, rotL, dir);
          fluxR = rotate_back(fluxR, rotR, dir);

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt * (lenL*fluxL - lenR*fluxR) / cellArea;
        
          // if (dir == IY && params.gravity) {
            
          //   //TODO: rendre plus propre...
          //   #if 1 // gravity toward (0,0)
          //     real_t cos, sin;
          //     {
          //       Pos pos = geometry.mapc2p_center(i, j);
          //       real_t norm = sqrt(pos[IX]*pos[IX] + pos[IY]*pos[IY]);
          //       cos = -pos[IX] / norm;
          //       sin = -pos[IY] / norm;
          //     }
          //     un_loc[IU] += dt * Q(j, i, IR) * params.g * cos;
          //     un_loc[IV] += dt * Q(j, i, IR) * params.g * sin;

          //   #else // gravity down
          //     un_loc[IV] += dt * Q(j, i, IR) * params.g;
          //   #endif

          //   un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
          // }

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
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
    if (params.gravity != GRAV_NONE)
      grav_functor.applyGravity(Q, Unew, dt);
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
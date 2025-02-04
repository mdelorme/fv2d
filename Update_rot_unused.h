#pragma once 

#include "SimInfo.h"
#include "RiemannSolvers_rot.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

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
}

class UpdateFunctor {
public:
  Params params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;
  Geometry geometry;

  Array slopesX, slopesY;

  UpdateFunctor(const Params &params)
    : params(params), bc_manager(params),
      tc_functor(params), visc_functor(params),
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
          Pos normL = geometry.interfaceRot(i, j, dir, &lenL);
          Pos normR = geometry.interfaceRot(i+dxp, j+dyp, dir, &lenR);

          real_t dLL, dLR, dRL, dRR;
          {
            Pos reconsLengthL = geometry.cellReconsLength(i, j, dir);
            Pos reconsLengthR = geometry.cellReconsLength(i+dxp, j+dyp, dir);
            dLL = reconsLengthL[0];
            dLR = reconsLengthL[1];
            dRL = reconsLengthR[0];
            dRR = reconsLengthR[1];
          }
          
          State qCL, qCR, qL, qR;
          {
            State qC = getStateFromArray(Q, i, j);
                  qL = getStateFromArray(Q, i+dxm, j+dym);
                  qR = getStateFromArray(Q, i+dxp, j+dyp);
              
            qCL = reconstruct(qC, slopes, i,     j,     -dLR, dir, params);
            qCR = reconstruct(qC, slopes, i,     j,      dRL, dir, params);
            qL  = reconstruct(qL, slopes, i+dxm, j+dym,  dLL, dir, params);
            qR  = reconstruct(qR, slopes, i+dxp, j+dyp, -dRR, dir, params);
          }

          /////////////// end Geometry

          // Calling the right Riemann solver
          auto riemann = [&](State qL, State qR, State &flux, real_t &pout, Pos rot) {
              hllc_rot(qL, qR, flux, pout, rot, params);
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL, normL);
          riemann(qCR, qR, fluxR, poutR, normR);

          fluxL = lenL * fluxL;
          fluxR = lenR * fluxR;

          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
            if (j==params.jbeg)
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*params.g*params.dy, 0.0};
            else 
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*params.g*params.dy, 0.0};
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(fluxL - fluxR) / cellArea;
        
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
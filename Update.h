#pragma once 

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"
#include "Gravity.h"
#include "Geometry.h"
#include "Gradient.h"

namespace fv2d {

namespace {
  using StatePair = Kokkos::pair<State, State>;

  /**
   * Reconstructs value at interface for axis-aligned interfaces
   * 
   * @param Q primitive variables array
   * @param slopes slope array to apply along current dir
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param length signed distance to interface
   * @param dir direction of the current axis
   * @param params global parameters of the run
   * @return reconstructed state at the interface
   */
  KOKKOS_INLINE_FUNCTION
  State reconstructCartesian(Array Q, Array slopes, int i, int j, real_t length, IDir dir, const DeviceParams &params, const Geometry &geometry) {
    State q     = getStateFromArray(Q, i, j);
    State slope = getStateFromArray(slopes, i, j);

    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * length; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = q[IP] + length * q[IR] * getGravity(i, j, params, geometry)[IY] * params.dy;
      default:  res = q; // Piecewise Constant
    }

    return res;
  }

  /**
   * Gradient based reconstruction
   * 
   * @param Q primitive variables array
   * @param slopesX gradient along x : dQ/dx
   * @param slopesY gradient along y : dQ/dy
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir direction of the current axis
   * @param side side of the interface
   * @param params global parameters of the run
   * @param geometry geometry object for transformations
   * @return the reconstructed state at interface applying the gradient
   */
  KOKKOS_INLINE_FUNCTION
  State reconstructGradient(Array Q, Array slopesX, Array slopesY, int i, int j, IDir dir, ISide side, const Geometry& geometry) {
    State q = getStateFromArray(Q, i, j);
    State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};
    Pos cf = geometry.getCenterToFace(i, j, dir, side);
    return q + cf[IX] * grad[IX] + cf[IY] * grad[IY];
  }

  /**
   * Reconstructs the states left and right of an interface
   * 
   * @param Q Primitive variables array
   * @param slopesX gradient along x : dQ/dx
   * @param slopesY gradient along y : dQ/dy
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interface
   * @param side side of the interface 
   * @param geometry geometry object for transformations
   * @return the left and right states reconstructed for the interface
   */
  KOKKOS_INLINE_FUNCTION
  StatePair reconstruct(Array Q, Array slopesX, Array slopesY, int i, int j, IDir dir, ISide side, const Geometry& geometry) {
    const auto &params = geometry.params;

    const int iL = i - (side == ILEFT  && dir == IX);
    const int jL = j - (side == ILEFT  && dir == IY);
    const int iR = i + (side == IRIGHT && dir == IX);
    const int jR = j + (side == IRIGHT && dir == IY);

    StatePair q;

    switch (params.reconstruction) {
      case PLM: 
      case PCM_WB: 
      case PCM:
      {
        auto& slopes = (dir == IX ? slopesX : slopesY);
        auto [dL, dR] = geometry.getCellReconsLenghts(iR, jR, dir);
        q.first  = reconstructCartesian(Q, slopes, iL, jL,  dL, dir, params, geometry);
        q.second = reconstructCartesian(Q, slopes, iR, jR, -dR, dir, params, geometry);
        break;
      }
      case RECONS_GRADIENT: // naive gradient reconstruction
      {
        q.first  = reconstructGradient(Q, slopesX, slopesY, iL, jL, dir, IRIGHT, geometry);
        q.second = reconstructGradient(Q, slopesX, slopesY, iR, jR, dir, ILEFT,  geometry);
        break;
      }
    }

    return q;
  }  
  
  /** 
   * Rotates a state from physical to logical space 
   * 
   * @param q the state to rotate
   * @param normal the normal vector of the face
   * @return the state where vector quantities have been rotated in logical space
  */
  KOKKOS_INLINE_FUNCTION
  State rotate(const State &q, const Pos &normal) {
    State res = q;
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t cos = normal[IX];
    const real_t sin = normal[IY];
  
    res[IU] =  cos * u + sin * v;
    res[IV] = -sin * u + cos * v;
    return res;
  }
   
  /** 
   * Rotates back the statr from logical to physical space 
   * 
   * @param q the state to rotate
   * @param normal the normal vector of the face
   * @return the state where vector quantities have been rotated back into physical space
  */ 
  KOKKOS_INLINE_FUNCTION
  State rotate_back(const State &q, const Pos &normal) {
    State res = q;
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t cos = normal[IX];
    const real_t sin = normal[IY];

    res[IU] =  cos * u - sin * v;
    res[IV] =  sin * u + cos * v;
    return res;
  }

}

class UpdateFunctor {
public:
  Params full_params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;
  GravityFunctor gravity_functor;

  Array slopesX, slopesY;

  Geometry geometry;

  UpdateFunctor(const Params &full_params)
    : full_params(full_params), bc_manager(full_params),
      tc_functor(full_params), visc_functor(full_params),
      gravity_functor(full_params),
      geometry(full_params.device_params)
       {
      auto device_params = full_params.device_params;
      slopesX = Array("SlopesX", device_params.Nty, device_params.Ntx, Nfields);
      slopesY = Array("SlopesY", device_params.Nty, device_params.Ntx, Nfields);
    };
  ~UpdateFunctor() = default;

  /**
   * Computes the slopes according to a given reconstruction
   * 
   * @param Q: Array of primitive variables on which slopes will be calculated
   */
  void computeSlopes(const Array &Q) const {
    auto &slopesX  = this->slopesX;
    auto &slopesY  = this->slopesY;
    auto &params   = this->full_params.device_params;
    auto &geometry = this->geometry; 

    switch (params.reconstruction) {
      case PLM:
        Kokkos::parallel_for(
          "Slopes", 
          full_params.range_slopes,
          KOKKOS_LAMBDA(const int i, const int j) {
            const real_t dL = geometry.getCellReconsLengthSlope(i,  j,  IX);
            const real_t dR = geometry.getCellReconsLengthSlope(i+1,j,  IX);
            const real_t dD = geometry.getCellReconsLengthSlope(i,  j,  IY);
            const real_t dU = geometry.getCellReconsLengthSlope(i,  j+1,IY);
            
            for (int ivar=0; ivar < Nfields; ++ivar) {
              real_t dqL = (Q(j, i, ivar)   - Q(j, i-1, ivar)) / dL;
              real_t dqR = (Q(j, i+1, ivar) - Q(j, i, ivar)  ) / dR;
              real_t dqD = (Q(j, i, ivar)   - Q(j-1, i, ivar)) / dD; 
              real_t dqU = (Q(j+1, i, ivar) - Q(j, i, ivar)  ) / dU; 

              auto minmod = [](real_t dqL, real_t dqR) -> real_t {
                if (dqL*dqR < 0.0)
                  return 0.0;
                else if (fabs(dqL) < fabs(dqR))
                  return dqL;
                else
                  return dqR;
              };

              slopesX(j, i, ivar) = minmod(dqL, dqR);
              slopesY(j, i, ivar) = minmod(dqD, dqU);
            }
          });
        break;
      case RECONS_GRADIENT:
        Kokkos::parallel_for(
          "Slopes", 
          full_params.range_slopes,
          KOKKOS_LAMBDA(const int i, const int j) {
            auto grad = computeGradient(Q, getStateFromArray, i, j, geometry);
            setStateInArray(slopesX, i, j, grad[IX]);
            setStateInArray(slopesY, i, j, grad[IY]);
          });
        break;
    }

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

  /**
   * Computes the fluxes and updates the cells
   * 
   * @param Q: Array of primitive variables 
   * @param Unew: Array of conservative variables that will be updated
   * @param dt: Timestep
   */
  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto &params   = full_params.device_params;
    auto &slopesX  = this->slopesX;
    auto &slopesY  = this->slopesY;
    auto &geometry = this->geometry;

    Kokkos::parallel_for(
      "Update", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        const real_t cellArea = geometry.cellArea(i,j);

        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          // 1. Calculate the rotation matrices for each interface
          auto [rotL, lenL] = geometry.getFaceInfo(i, j, dir, ILEFT);
          auto [rotR, lenR] = geometry.getFaceInfo(i, j, dir, IRIGHT);

          // 2. Reconstructs the values at each interface
          auto [qL, qCL] = reconstruct(Q, slopesX, slopesY, i, j, dir, ILEFT,  geometry);
          auto [qCR, qR] = reconstruct(Q, slopesX, slopesY, i, j, dir, IRIGHT, geometry);

          /**
           * Note : 
           * On original geometry branch, there was a fix here for 
           * fixing the value of the state at boundary before calling 
           * the Riemann Solver
           * 
           * https://github.com/mdelorme/fv2d/blob/geometry/Update.h#L359
           **/

          // Calling the right Riemann solver
          auto riemann = [&](State qL, State qR, State &flux, Pos &rot, real_t &pout) {
            qL = rotate(qL, rot);
            qR = rotate(qR, rot);

            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params); break;
              default: hllc(qL, qR, flux, pout, params); break;
            }

            flux = rotate_back(flux, rot);
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, rotL, poutL);
          riemann(qCR, qR, fluxR, rotR, poutR);

          /**
           * Note: Here, we should have the flux overriding for boundaries
           */

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(lenL*fluxL - lenR*fluxR) / cellArea;

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);
      });
  }

  void euler_step(Array Q, Array Unew, real_t dt) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hypperbolic udpate
    if (full_params.device_params.reconstruction == PLM)
      computeSlopes(Q);
    computeFluxesAndUpdate(Q, Unew, dt);

    // Splitted terms
    if (full_params.device_params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (full_params.device_params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
    if (full_params.device_params.gravity_mode != GRAV_NONE)
      gravity_functor.applyGravity(Q, Unew, dt);
  }

  void update(Array Q, Array Unew, real_t dt) {
    if (full_params.time_stepping == TS_EULER)
      euler_step(Q, Unew, dt);
    else if (full_params.time_stepping == TS_RK2) {
      auto params = full_params.device_params;
      Array U0    = Array("U0", params.Nty, params.Ntx, Nfields);
      Array Ustar = Array("Ustar", params.Nty, params.Ntx, Nfields);
      
      // Step 1
      Kokkos::deep_copy(U0, Unew);
      Kokkos::deep_copy(Ustar, Unew);
      euler_step(Q, Ustar, dt);
      
      // Step 2
      Kokkos::deep_copy(Unew, Ustar);
      consToPrim(Ustar, Q, full_params);
      euler_step(Q, Unew, dt);

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
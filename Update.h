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
  State reconstruct1D(Array Q, Array slopes, int i, int j, real_t length, IDir dir, const Params &params) {
    State q     = getStateFromArray(Q, i, j);
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
  }

  // Gradient based reconstruction with Bruner's limiter
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<State, 2> reconstructBruner(Array Q, Array slopesX, Array slopesY, int iL, int jL, int iR, int jR, const Geometry& geometry, const Params &params) {
    State qL = getStateFromArray(Q, iL, jL);
    State qR = getStateFromArray(Q, iR, jR);
    State gradL[2] = {getStateFromArray(slopesX, iL, jL), getStateFromArray(slopesY, iL, jL)};
    State gradR[2] = {getStateFromArray(slopesX, iR, jR), getStateFromArray(slopesY, iR, jR)};
    constexpr State one = {1.0, 1.0, 1.0, 1.0};

    Pos rLR = geometry.mapc2p_center(iR,jR) - geometry.mapc2p_center(iL,jL);
    State rfL = 2 * (gradL[IX] * rLR[IX] + gradL[IY] * rLR[IY]) / (qR - qL) - one;
    State rfR = 2 * (gradR[IX] * rLR[IX] + gradR[IY] * rLR[IY]) / (qR - qL) - one;

    auto minmod = [](real_t r){
      r = ((r < 1) ? r : 1);
      r = ((r > 0) ? r : 0);
      return r;
    };

    State limitedL, limitedR;
    for (int ivar=0; ivar < Nfields; ++ivar) {
      real_t psiL = minmod(rfL[ivar]);
      real_t psiR = minmod(rfR[ivar]);
      // printf("r = %.2lf / %.2lf\n", psiL, psiR);
      real_t dq = qR[ivar] - qL[ivar];
      limitedL[ivar] = qL[ivar] + 0.5 * psiL * dq;
      limitedR[ivar] = qR[ivar] - 0.5 * psiR * dq;
    }

    return {limitedL, limitedR};
  }

  // Gradient based reconstruction with Barth and Jesperson scheme
  KOKKOS_INLINE_FUNCTION
  State reconstructBJ(Array Q, Array slopesX, Array slopesY, Array psi_in, int i, int j, IDir dir, ISide side, const Geometry& geometry, const Params &params) {
    State q = getStateFromArray(Q, i, j);
    State psi = getStateFromArray(psi_in, i, j);
    State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};
    Pos cf = geometry.centerToFace(i, j, dir, side);
    return q + psi * (cf[IX] * grad[IX] + cf[IY] * grad[IY]);
  }

  // Gradient based naïve reconstruction
  KOKKOS_INLINE_FUNCTION
  State reconstructNaive(Array Q, Array slopesX, Array slopesY, int i, int j, IDir dir, ISide side, const Geometry& geometry, const Params &params) {
    State q = getStateFromArray(Q, i, j);
    State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};
    Pos cf = geometry.centerToFace(i, j, dir, side);
    return q + cf[IX] * grad[IX] + cf[IY] * grad[IY];
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<State, 2> reconstruct(Array Q, Array slopesX, Array slopesY, Array psi, int i, int j, IDir dir, ISide side, const Geometry& geometry, const Params &params) {
    int sx = (dir == IX ? 1 : 0);
    int sy = (dir == IY ? 1 : 0);
    int iL = i - (side == ILEFT ? sx : 0 );
    int jL = j - (side == ILEFT ? sy : 0 );
    int iR = i + (side == ILEFT ?  0 : sx);
    int jR = j + (side == ILEFT ?  0 : sy);
    // int iL = i - (side == ILEFT  && dir == IX);
    // int jL = j - (side == ILEFT  && dir == IY);
    // int iR = i + (side == IRIGHT && dir == IX);
    // int jR = j + (side == IRIGHT && dir == IY);

    Kokkos::Array<State, 2> q;

    switch (params.reconstruction) {
      case PLM: case PCM_WB: case PCM: // 1d reconstruction
      {
        auto& slopes = (dir == IX ? slopesX : slopesY);
        auto [dL, dR] = geometry.cellReconsLength(i, j, dir);
        q[ILEFT]  = reconstruct1D(Q, slopes, iL, jL,  dL, dir, params);
        q[IRIGHT] = reconstruct1D(Q, slopes, iR, jR, -dR, dir, params);
        break;
      }
      case RECONS_NAIVE: // naive gradient reconstruction
      {
        q[ILEFT]  = reconstructNaive(Q, slopesX, slopesY, iL, jL, dir, IRIGHT, geometry, params);
        q[IRIGHT] = reconstructNaive(Q, slopesX, slopesY, iR, jR, dir, ILEFT,  geometry, params);
        break;
      }
      case RECONS_BRUNER: // limited gradient reconstruction
        q = reconstructBruner(Q, slopesX, slopesY, iL, jL, iR, jR, geometry, params);
        break;
      case RECONS_BJ:
      {
        q[ILEFT]  = reconstructBJ(Q, slopesX, slopesY, psi, iL, jL, dir, IRIGHT, geometry, params);
        q[IRIGHT] = reconstructBJ(Q, slopesX, slopesY, psi, iR, jR, dir, ILEFT,  geometry, params);
        break;
      }
    }

    return q;
  }

  KOKKOS_INLINE_FUNCTION
  State rotate(const State &q, const Pos &normale) {
    State res = q;
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t cos = normale[IX];
    const real_t sin = normale[IY];
  
    res[IU] =  cos * u + sin * v;
    res[IV] = -sin * u + cos * v;
    return res;
  }
  
  KOKKOS_INLINE_FUNCTION
  State rotate_back(const State &q, const Pos &normale) {
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
  Array psi;

  UpdateFunctor(const Params &params)
    : params(params), bc_manager(params),
      tc_functor(params), visc_functor(params), grav_functor(params),
      geometry(params) {
      
      slopesX = Array("SlopesX", params.Nty, params.Ntx, Nfields);
      slopesY = Array("SlopesY", params.Nty, params.Ntx, Nfields);
      if (params.reconstruction == RECONS_BJ)
        psi = Array("psi(BJ limiter)", params.Nty, params.Ntx, Nfields);
    };
  ~UpdateFunctor() = default;

  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto params  = this->params;
    auto &geometry = this->geometry;
    
    switch (params.reconstruction) {
      case PCM: 
        break;
      case PLM: case PCM_WB: // 1d slopes
        Kokkos::parallel_for(
          "Slopes",
          params.range_slopes,
          KOKKOS_LAMBDA(const int i, const int j) {
            const real_t dL = geometry.cellReconsLengthSlope(i,  j,  IX);
            const real_t dR = geometry.cellReconsLengthSlope(i+1,j,  IX);
            const real_t dU = geometry.cellReconsLengthSlope(i,  j,  IY);
            const real_t dD = geometry.cellReconsLengthSlope(i,  j+1,IY);
            
            for (int ivar=0; ivar < Nfields; ++ivar) {
              real_t dqL = (Q(j, i, ivar)   - Q(j, i-1, ivar)) / dL;
              real_t dqR = (Q(j, i+1, ivar) - Q(j, i, ivar)  ) / dR;
              real_t dqU = (Q(j, i, ivar)   - Q(j-1, i, ivar)) / dU; 
              real_t dqD = (Q(j+1, i, ivar) - Q(j, i, ivar)  ) / dD; 

              auto minmod = [](real_t dqL, real_t dqR) -> real_t {
                if (dqL*dqR < 0.0)
                  return 0.0;
                else if (fabs(dqL) < fabs(dqR))
                  return dqL;
                else
                  return dqR;
              };

              slopesX(j, i, ivar) = minmod(dqL, dqR);
              slopesY(j, i, ivar) = minmod(dqU, dqD);
            }
          });
          break;
          
          case RECONS_NAIVE: case RECONS_BRUNER: case RECONS_BJ:  
          Kokkos::parallel_for(
          "Slopes(gradient)",
          params.range_slopes,
          KOKKOS_LAMBDA(const int i, const int j) {
            auto grad = computeGradient(Q, getStateFromArray, i, j, geometry, params.gradient_type);
            setStateInArray(slopesX, i, j, grad[IX]);
            setStateInArray(slopesY, i, j, grad[IY]);
          });
          break;
          default:
          {
            Kokkos::abort("unknown reconstruction reconstruction parameter");
          }
    }
  }

  void computeLimiterBJ(const Array &Q) {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto psi_glob = this->psi;
    auto &geometry = this->geometry;

    Kokkos::parallel_for(
      "BJ limiter",
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        State q = getStateFromArray(Q, i, j);
        State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};

        auto min = [](real_t a, real_t b){ return (a < b) ? a : b;};
        auto max = [](real_t a, real_t b){ return (a > b) ? a : b;};
        auto lim = [](real_t x){ return (x < 1) ? x : 1;};
        constexpr real_t inf = Kokkos::Experimental::infinity_v<real_t>;

        State dqmin = { inf,  inf,  inf,  inf}, 
              dqmax = {-inf, -inf, -inf, -inf};
        for (IDir dir : {IX, IY}) {
          for (ISide side : {ILEFT, IRIGHT}) {
            int iN = i + (dir == IX ? (side == ILEFT ? -1 : 1) : 0 );
            int jN = j + (dir == IY ? (side == ILEFT ? -1 : 1) : 0 );
            State qN = getStateFromArray(Q, iN, jN);
            for (int ivar=0; ivar < Nfields; ++ivar) {
              dqmin[ivar] = min(qN[ivar] - q[ivar], dqmin[ivar]);
              dqmax[ivar] = max(qN[ivar] - q[ivar], dqmax[ivar]);
            }
          }
        }

        State psi = {1, 1, 1, 1};
        for (IDir dir : {IX, IY}) {
          for (ISide side : {ILEFT, IRIGHT}) {
            Pos rf = geometry.centerToFace(i, j, dir, side);
            State dqf = grad[IX] * rf[IX] + grad[IY] * rf[IY];

            // int iN = i + (dir == IX ? (side == ILEFT ? -1 : 1) : 0 );
            // int jN = j + (dir == IY ? (side == ILEFT ? -1 : 1) : 0 );
            // State qN = getStateFromArray(Q, iN, jN);

            // if (dqf[IR] != 0.0 || dqf[IU] != 0.0 || dqf[IV] != 0.0 || dqf[IP] != 0.0)
            // printf("% .2f % .2f % .2f % .2f\n", dqf[IR], dqf[IU], dqf[IV], dqf[IP]);
            
            for (int ivar=0; ivar < Nfields; ++ivar) {
              real_t psi_f;
              if (dqf[ivar] > 0) {
                // psi_f = (max(q[ivar], qN[ivar]) - q[ivar]) / dqf[ivar];
                psi_f = dqmax[ivar] / dqf[ivar];
              }
              else if (dqf[ivar] < 0) {
                // psi_f = (min(q[ivar], qN[ivar]) - q[ivar]) / dqf[ivar];
                // psi_f = (q[ivar] - min(q[ivar], qN[ivar])) / dqf[ivar];
                psi_f = dqmin[ivar] / dqf[ivar];
                // printf("%.2f\n", psi_f);
              }
              else {
                psi_f = 1;
              }
              psi_f = lim(psi_f);
              psi[ivar] = min(psi_f, psi[ivar]);
            }
          }
        }
        // if (psi[IR] != 1.0 || psi[IU] != 1.0 || psi[IV] != 1.0 || psi[IP] != 1.0)
        // printf("%.2f %.2f %.2f %.2f\n", psi[IR], psi[IU], psi[IV], psi[IP]);
        setStateInArray(psi_glob, i, j, psi);
      });
  }

  void computeHancockStep(const Array &Q, real_t dt) const {
    auto params  = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    const real_t dt_half = 0.5 * dt;
    const real_t gamma = params.gamma0;

    Kokkos::parallel_for(
      "MUSCL-Hancock",
      params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        auto [ r,   u,   v,   p ] = getStateFromArray(Q, i, j);
        auto [drx, dux, dvx, dpx] = getStateFromArray(slopesX, i, j);
        auto [dry, duy, dvy, dpy] = getStateFromArray(slopesY, i, j);
        
        Q(j, i, IR) = r - dt_half * (u * drx + r * dux          +  v * dry + r * dvy)        ;
        Q(j, i, IU) = u - dt_half * (u * dux + dpx / r          +  v * duy)                  ;
        Q(j, i, IV) = v - dt_half * (u * dvx                    +  v * dvy + dpy / r)        ;
        Q(j, i, IP) = p - dt_half * (gamma * p * dux + u * dpx  +  gamma * p * dvy + v * dpy);
      });
  }
  
  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto psi = this->psi;
    auto &geometry = this->geometry;

    Kokkos::parallel_for(
      "Update", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        const real_t cellArea = geometry.cellArea(i,j);
        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          real_t lenL, lenR;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          Pos rotR = geometry.getRotationMatrix(i, j, dir, IRIGHT, lenR);

          auto [qL, qCL] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, ILEFT,  geometry, params);
          auto [qCR, qR] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, IRIGHT, geometry, params);

          // Calling the right Riemann solver
          auto riemann = [&](State &qL, State &qR, State &flux, Pos &rot, real_t &pout) {
            qL = rotate(qL , rot);
            qR = rotate(qR , rot);
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

		  auto get_dP_wb = [&](ISide side) {
            const Pos center = geometry.mapc2p_center(i,j);
            const real_t r = norm(center);
            // const Pos center_to_face = geometry.centerToFace(i,j,dir,side);
            const Pos face_to_face = geometry.faceToFace(i,j,dir);
            // return 0;
            // std::string sideinfo; sideinfo += (dir==IX?'X':'Y');sideinfo += (side==ILEFT?'L':'R');
            // sideinfo += "  ";
            // if (dir == IY && side == IRIGHT)
            // std::cout << sideinfo << '('<< i << ',' << j << ")\t"
            // << dot(center, center_to_face) / r << "\t"
            // << norm(geometry.faceCenter(i,j,dir,side)) << std::endl;
            // const real_t dP = params.spl_grav.GetValue(r) * Q(j, i, IR) * dot(center, center_to_face) / r;
            const real_t dP = params.spl_grav.GetValue(r) * Q(j, i, IR) * dot(center, face_to_face) / r;
            // std::cout << (dir==IX?'X':'Y') << (side==ILEFT?'L':'R') << ": " << dP << std::endl;
            // return params.spl_grav.GetValue(r) * qC[IR] * dot(center, center_to_face) / r;
            // return dP;
            return dP * (side==IRIGHT?-1.0:1.0);
          };

		  if (params.reflective_flux) {
		    if (dir == IX) {
              if (i==params.ibeg) {
				real_t dP = (params.reflective_flux_wb) ? get_dP_wb(ILEFT) : 0;
                fluxL = rotate_back(State{0.0, poutL - dP, 0.0, 0.0}, rotL);
			  }
              else if (i==params.iend-1) {
				real_t dP = (params.reflective_flux_wb) ? get_dP_wb(IRIGHT) : 0;
                fluxR = rotate_back(State{0.0, poutR - dP, 0.0, 0.0}, rotR);
              }
			}
            else {
              if (j==params.jbeg) {
				real_t dP = (params.reflective_flux_wb) ? get_dP_wb(ILEFT) : 0;
                fluxL = rotate_back(State{0.0, poutL - dP, 0.0, 0.0}, rotL);
			  }
              else if (j==params.jend-1) {
				real_t dP = (params.reflective_flux_wb) ? get_dP_wb(IRIGHT) : 0;
                fluxR = rotate_back(State{0.0, poutR - dP, 0.0, 0.0}, rotR);
			  }
            }
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt * (lenL*fluxL - lenR*fluxR) / cellArea;

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        if (params.use_pressure_gradient) // crash imédiatement avec HLLC, ~ok avec HLL
        {
          auto getState = [](const Array& Q, int i, int j){return Q(j,i,IP);};
          Kokkos::Array<real_t, 2> grad = computeGradient(Q, getState, i, j, geometry, params.gradient_type);
          Unew(j, i, IU) -= dt * grad[IX];
          Unew(j, i, IV) -= dt * grad[IY];
        }

        // Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
      });
  }

  // OLD (contient flux reflectif & wb)
#if 0
  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto &geometry = this->geometry;

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

          real_t lenL, lenR;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          Pos rotR = geometry.getRotationMatrix(i, j, dir, IRIGHT, lenR);

          auto [dLL, dLR] = geometry.cellReconsLength(i, j, dir);
          auto [dRL, dRR] = geometry.cellReconsLength(i+dxp, j+dyp, dir);
          
          State qL  = reconstruct(Q, slopes, i+dxm, j+dym,  dLL, dir, params);
          State qCL = reconstruct(Q, slopes, i,     j,     -dLR, dir, params);
          State qCR = reconstruct(Q, slopes, i,     j,      dRL, dir, params);
          State qR  = reconstruct(Q, slopes, i+dxp, j+dyp, -dRR, dir, params);

          qL  = rotate(qL , rotL);
          qCL = rotate(qCL, rotL);
          qCR = rotate(qCR, rotR);
          qR  = rotate(qR , rotR);

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
          #if 0 // well balancing à réparer
          auto get_dP_wb = [&](ISide side) {
              const Pos center = geometry.mapc2p_center(i,j);
              const real_t r = norm(center);
              // const Pos center_to_face = geometry.centerToFace(i,j,dir,side);
              const Pos face_to_face = geometry.faceToFace(i,j,dir);
              // return 0;
              // std::string sideinfo; sideinfo += (dir==IX?'X':'Y');sideinfo += (side==ILEFT?'L':'R'); 
              // sideinfo += "  ";
              // if (dir == IY && side == IRIGHT)
              // std::cout << sideinfo << '('<< i << ',' << j << ")\t"
              // << dot(center, center_to_face) / r << "\t" 
              // << norm(geometry.faceCenter(i,j,dir,side)) << std::endl;
              // const real_t dP = params.spl_grav.GetValue(r) * Q(j, i, IR) * dot(center, center_to_face) / r;
              const real_t dP = params.spl_grav.GetValue(r) * Q(j, i, IR) * dot(center, face_to_face) / r;
              // std::cout << (dir==IX?'X':'Y') << (side==ILEFT?'L':'R') << ": " << dP << std::endl;
              // return params.spl_grav.GetValue(r) * qC[IR] * dot(center, center_to_face) / r;
              return dP * (side==IRIGHT?-1.0:1.0);
              };
          
          if (dir == IX) {
            if (i==params.ibeg)
              fluxL = State{0.0, poutR + get_dP_wb(ILEFT),  0.0, 0.0};
            else if (i==params.iend-1)
              fluxR = State{0.0, poutL + get_dP_wb(IRIGHT), 0.0, 0.0};
          }
          else {
            if (j==params.jbeg)
              fluxL = State{0.0, poutR + get_dP_wb(ILEFT),  0.0, 0.0};
            else if (j==params.jend-1)
              fluxR = State{0.0, poutL + get_dP_wb(IRIGHT), 0.0, 0.0};
          }
          #else

          // if (dir == IX) {
          //   if (i==params.ibeg)
          //     fluxL = State{0.0, poutL,  0.0, 0.0};
          //   else if (i==params.iend-1)
          //     fluxR = State{0.0, poutR, 0.0, 0.0};
          // }
          // else {
          //   if (j==params.jbeg)
          //     fluxL = State{0.0, poutL,  0.0, 0.0};
          //   else if (j==params.jend-1)
          //     fluxR = State{0.0, poutR, 0.0, 0.0};
          // }
          #endif

          fluxL = rotate_back(fluxL, rotL);
          fluxR = rotate_back(fluxR, rotR);

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt * (lenL*fluxL - lenR*fluxR) / cellArea;

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        if (params.use_pressure_gradient) // crash imédiatement avec HLLC, ~ok avec HLL
        {
          auto getState = [](const Array& Q, int i, int j){return Q(j,i,IP);};
          Kokkos::Array<real_t, 2> grad = computeGradient(Q, getState, i, j, geometry, params.gradient_type);
          Unew(j, i, IU) -= dt * grad[IX];
          Unew(j, i, IV) -= dt * grad[IY];
        }

        // Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
      });
  }
#endif

  void computeFluxesAndUpdate_CTU(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto psi = this->psi;
    auto &geometry = this->geometry;

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
          auto [qL, qR] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, ILEFT, geometry, params);
          // State qL = reconstruct(Q, slopes, i+dxm, j+dym,  1.0, dir, params);
          // State qR = reconstruct(Q, slopes, i,     j,     -1.0, dir, params);

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
          // TODO CHANGE DX DY ...
          real_t dtddir  = dt/(dir == IX ? params.dx : params.dy);
          real_t dtdtdir = dt/(dir == IY ? params.dx : params.dy);
          IDir tdir = (dir == IX) ? IY : IX; // 2d case

          auto reconstructTransverse = [&](int ii, int jj, real_t sign) {
            State U = reconstruct1D(Q, slopes, ii, jj, sign, dir, params);
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
    
    computeSlopes(Q);
    if (params.reconstruction == RECONS_BJ)
      computeLimiterBJ(Q);
    if (params.hancock_ts)
      computeHancockStep(Q, dt);

    // Hyperbolic udpate
    switch(params.flux_solver) { 
      case FLUX_GODOUNOV: computeFluxesAndUpdate(Q, Unew, dt); break;
      case FLUX_CTU:      computeFluxesAndUpdate_CTU(Q, Unew, dt); break;
    }

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

#pragma once 

#include "Geometry.h"
#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"
#include "BodyForces.h"

namespace fv2d {

namespace {
  KOKKOS_INLINE_FUNCTION
  State reconstruct1D(Array Q, Array slopes, int i, int j, real_t length, IDir dir, const DeviceParams &params) {
    State q     = getStateFromArray(Q, i, j);
    State slope = getStateFromArray(slopes, i, j);
    
    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * length; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = q[IP] + length * q[IR] * params.gy; // getGravity(i, j, dir, params)
      default:  res = q; // Piecewise Constant
    }
    return res;
  }

  // Gradient based reconstruction with Bruner's limiter
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<State, 2> reconstructBruner(Array Q, Array slopesX, Array slopesY, int iL, int jL, int iR, int jR, const Geometry& geometry, const DeviceParams &params) {
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
  State reconstructBJ(Array Q, Array slopesX, Array slopesY, Array psi_in, int i, int j, IDir dir, ISide side, const Geometry& geometry, const DeviceParams &params) {
    State q = getStateFromArray(Q, i, j);
    State psi = getStateFromArray(psi_in, i, j);
    State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};
    Pos cf = geometry.centerToFace(i, j, dir, side);
    return q + psi * (cf[IX] * grad[IX] + cf[IY] * grad[IY]);
  }

  // Gradient based naïve reconstruction
  KOKKOS_INLINE_FUNCTION
  State reconstructNaive(Array Q, Array slopesX, Array slopesY, int i, int j, IDir dir, ISide side, const Geometry& geometry, const DeviceParams &params) {
    State q = getStateFromArray(Q, i, j);
    State grad[2] = {getStateFromArray(slopesX, i, j), getStateFromArray(slopesY, i, j)};
    Pos cf = geometry.centerToFace(i, j, dir, side);
    return q + cf[IX] * grad[IX] + cf[IY] * grad[IY];
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<State, 2> reconstruct(Array Q, Array slopesX, Array slopesY, Array psi, int i, int j, IDir dir, ISide side, const Geometry& geometry, const DeviceParams &params) {
    int iL = i - (side == ILEFT  && dir == IX);
    int jL = j - (side == ILEFT  && dir == IY);
    int iR = i + (side == IRIGHT && dir == IX);
    int jR = j + (side == IRIGHT && dir == IY);

    real_t r = norm(geometry.faceCenter(i, j, dir, side));
    const real_t rho0 = params.spl_rho(r);
    const real_t prs0 = params.spl_prs(r);  

    Kokkos::Array<State, 2> q;

    switch (params.reconstruction) {
      case PLM: case PCM_WB: case PCM: // 1d reconstruction
      {
        auto& slopes = (dir == IX ? slopesX : slopesY);
        auto [dL, dR] = geometry.cellReconsLength(iR, jR, dir);
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
    q[ILEFT][IR]  = q[ILEFT][IR]  * rho0;
    q[ILEFT][IP]  = q[ILEFT][IP]  * prs0;
    q[IRIGHT][IR] = q[IRIGHT][IR] * rho0;
    q[IRIGHT][IP] = q[IRIGHT][IP] * prs0;
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
  Params full_params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;
  GravityFunctor grav_functor;
  CoriolisFunctor coriolis_functor;
  HeatingFunctor heating_functor;
  Geometry geometry;

  Array slopesX, slopesY;
  Array psi;

  UpdateFunctor(const Params &full_params)
    : full_params(full_params),     bc_manager(full_params),
      tc_functor(full_params),      visc_functor(full_params), 
      grav_functor(full_params),    coriolis_functor(full_params),
      heating_functor(full_params), geometry(full_params.device_params) {
      
      auto device_params = full_params.device_params;
      slopesX = Array("SlopesX", device_params.Nty, device_params.Ntx, Nfields);
      slopesY = Array("SlopesY", device_params.Nty, device_params.Ntx, Nfields);
      if (device_params.reconstruction == RECONS_BJ)
        psi = Array("psi(BJ limiter)", device_params.Nty, device_params.Ntx, Nfields);
    };
  ~UpdateFunctor() = default;

  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto full_params  = this->full_params;
    auto params  = full_params.device_params;
    auto &geometry = this->geometry;
    
    switch (params.reconstruction) {
      case PCM: 
        break;
      case PLM: case PCM_WB: // 1d slopes
        Kokkos::parallel_for(
          "Slopes",
          full_params.range_slopes,
          KOKKOS_LAMBDA(const int i, const int j) {
            const real_t dL = geometry.cellReconsLengthSlope(i,  j,  IX);
            const real_t dR = geometry.cellReconsLengthSlope(i+1,j,  IX);
            const real_t dD = geometry.cellReconsLengthSlope(i,  j,  IY);
            const real_t dU = geometry.cellReconsLengthSlope(i,  j+1,IY);
            
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
          
          case RECONS_NAIVE: case RECONS_BRUNER: case RECONS_BJ:  
          Kokkos::parallel_for(
            "Slopes(gradient)",
            full_params.range_slopes,
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
    auto full_params  = this->full_params;
    auto psi_glob = this->psi;
    auto &geometry = this->geometry;

    Kokkos::parallel_for(
      "BJ limiter",
      full_params.range_dom,
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
    auto full_params  = this->full_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    const real_t dt_half = 0.5 * dt;
    auto dparams = this->full_params.device_params;
    const real_t gamma = dparams.gamma0;
    auto geometry = this->geometry;

    // not compatible with plm + non-cartesian

    Kokkos::parallel_for(
      "MUSCL-Hancock",
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        auto [ r,   u,   v,   p ] = getStateFromArray(Q, i, j);
        auto [drx, dux, dvx, dpx] = getStateFromArray(slopesX, i, j);
        auto [dry, duy, dvy, dpy] = getStateFromArray(slopesY, i, j);
        
        Pos u_r = geometry.mapc2p_center(i,j);
        const real_t radius = norm(u_r);
        u_r = u_r / radius;
        const real_t r0 = dparams.spl_rho(radius);
        const real_t p0 = dparams.spl_prs(radius);
        const real_t dr0 = dparams.spl_rho.GetDerivative(radius);
        const real_t dp0 = dparams.spl_prs.GetDerivative(radius);

        drx = drx + r / r0 * dr0 * u_r[IX];
        dry = dry + r / r0 * dr0 * u_r[IY];
        dpx = dpx + p / p0 * dp0 * u_r[IX];
        dpy = dpy + p / p0 * dp0 * u_r[IY];

        const real_t fac = p0 / r0; // facteur de correction, valide ???
        
        Q(j, i, IR) = r - dt_half * (u * drx + r * dux          +  v * dry + r * dvy)        ;
        Q(j, i, IU) = u - dt_half * (u * dux + fac * dpx / r    +  v * duy)                  ;
        Q(j, i, IV) = v - dt_half * (u * dvx                    +  v * dvy + fac * dpy / r)  ;
        Q(j, i, IP) = p - dt_half * (gamma * p * dux + u * dpx  +  gamma * p * dvy + v * dpy);
      });
  }
  
  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = full_params.device_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto psi = this->psi;
    auto &geometry = this->geometry;

    Kokkos::parallel_for(
      "Update", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        const real_t cellArea = geometry.cellArea(i,j);
        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          real_t lenL, lenR;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          Pos rotR = geometry.getRotationMatrix(i, j, dir, IRIGHT, lenR);

          auto [qL, qCL] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, ILEFT,  geometry, params);
          auto [qCR, qR] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, IRIGHT, geometry, params);
          
          if (params.fixed_spline_boundaries) {
            // Pos u_r = geometry.mapc2p_center(i,j);
            // const real_t r = norm(u_r);
            // u_r = u_r / r;
            // const real_t normal_vel = Q(j, i, IU) * u_r[IX] + Q(j, i, IV) * u_r[IY];

            if (dir==IX && (i==params.ibeg || i==params.iend-1)) { 
              State &q = (i==params.ibeg) ? qL : qR;
              Pos u_r = geometry.faceCenter(i,j, dir, (i==params.ibeg) ? ILEFT : IRIGHT);
              const real_t r = norm(u_r);
              u_r = u_r / r;
              const real_t normal_vel = Q(j, i, IU) * u_r[IX] + Q(j, i, IV) * u_r[IY];

              q[IR] = params.spl_rho(r);
              if (params.zero_velocity_boundary) {
                q[IU] = Q(j, i, IU) - 2 * normal_vel * u_r[IX];
                q[IV] = Q(j, i, IV) - 2 * normal_vel * u_r[IY];
              }
              else {
                q[IU] = 0;
                q[IV] = 0;
              }
              q[IP] = params.spl_prs(r);
            }
            if (dir==IY && (j==params.jbeg || j==params.jend-1)) { 
              State &q = (j==params.jbeg) ? qL : qR;
              Pos u_r = geometry.faceCenter(i,j, dir, (j==params.jbeg) ? ILEFT : IRIGHT);
              const real_t r = norm(u_r);
              u_r = u_r / r;
              const real_t normal_vel = Q(j, i, IU) * u_r[IX] + Q(j, i, IV) * u_r[IY];

              q[IR] = params.spl_rho(r);
              if (params.zero_velocity_boundary) {
                q[IU] = Q(j, i, IU) - 2 * normal_vel * u_r[IX];
                q[IV] = Q(j, i, IV) - 2 * normal_vel * u_r[IY];
              }
              else {
                q[IU] = 0;
                q[IV] = 0;
              }
              q[IP] = params.spl_prs(r);
            }
          }

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

          if (params.reflective_flux || params.reflective_flux_wb)
          {
            // reflective flux ne marche pas bien ==> vitesse radial très élevé au bords dès les premières itérations
            const real_t p0 = params.spl_prs(params.radial_radius);  

            if (dir == IX) {
              if (i==params.ibeg)
                fluxL = rotate_back(State{0.0, p0, 0.0, 0.0}, rotL);
              else if (i==params.iend-1)
                fluxR = rotate_back(State{0.0, p0, 0.0, 0.0}, rotR);
            }
            else {
              if (j==params.jbeg)
                fluxL = rotate_back(State{0.0, p0, 0.0, 0.0}, rotL);
              else if (j==params.jend-1)
                fluxR = rotate_back(State{0.0, p0, 0.0, 0.0}, rotR);
            }
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt * (lenL*fluxL - lenR*fluxR) / cellArea;

          setStateInArray(Unew, i, j, un_loc);
        };
        
        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);

        { // alpha beta scheme 
          real_t len;
          Pos lenL = geometry.getRotationMatrix(i, j, IX, ILEFT,  len); lenL = lenL * len;
          Pos lenR = geometry.getRotationMatrix(i, j, IX, IRIGHT, len); lenR = lenR * len;
          Pos lenD = geometry.getRotationMatrix(i, j, IY, ILEFT,  len); lenD = lenD * len;
          Pos lenU = geometry.getRotationMatrix(i, j, IY, IRIGHT, len); lenU = lenU * len;
          real_t r  = norm(geometry.mapc2p_center(i,j));
          real_t rl = norm(geometry.faceCenter(i, j, IX, ILEFT));
          real_t rr = norm(geometry.faceCenter(i, j, IX, IRIGHT));
          real_t rd = norm(geometry.faceCenter(i, j, IY, ILEFT));
          real_t ru = norm(geometry.faceCenter(i, j, IY, IRIGHT));
          const real_t prs0   = Q(j, i, IP);
          const real_t prs0l = params.spl_prs(rl);
          const real_t prs0r = params.spl_prs(rr);
          const real_t prs0d = params.spl_prs(rd);
          const real_t prs0u = params.spl_prs(ru);

          const real_t sx = prs0 * ( lenR[IX] * prs0r - lenL[IX] * prs0l + lenU[IX] * prs0u - lenD[IX] * prs0d) / cellArea;
          const real_t sy = prs0 * ( lenR[IY] * prs0r - lenL[IY] * prs0l + lenU[IY] * prs0u - lenD[IY] * prs0d) / cellArea; 
          Unew(j, i, IU) += dt * sx;
          Unew(j, i, IV) += dt * sy;
          Unew(j, i, IE) += dt * (Q(j, i, IU) * sx + Q(j, i, IV) * sy);
        }

        // Unew(j, i, IR) = fmax(1.0e-6, Unew(j, i, IR));
      });
  }

  void computeFluxesAndUpdate_CTU(Array Q, Array Unew, real_t dt) const {
    auto full_params = this->full_params;
    auto params = full_params.device_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto psi = this->psi;
    auto &geometry = this->geometry;

    Array Fhat[2] = { Array("Fhat_x", params.Nty, params.Ntx, Nfields),
                      Array("Fhat_y", params.Nty, params.Ntx, Nfields) };

    // predictor
    Kokkos::parallel_for(
      "Update CTU predictor", 
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {

        // Calling the right Riemann solver
        auto riemann = [&](State qL, State qR, State &flux, Pos &rot, real_t &pout) {
          qL = rotate(qL , rot);
          qR = rotate(qR , rot);
          switch (params.riemann_solver) {
            case HLL: hll(qL, qR, flux, pout, params); break;
            default: hllc(qL, qR, flux, pout, params); break;
          }
          flux = rotate_back(flux, rot);
        };

        // Lambda to update the cell along a direction
        auto updatePredictor = [&](int i, int j, IDir dir) {
          auto [qL, qR] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, ILEFT, geometry, params);

          real_t lenL;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          
          // Calculating flux "hat" left
          State  fluxL;
          real_t pout;

          riemann(qL, qR, fluxL, rotL, pout);
          setStateInArray(Fhat[dir], i, j, fluxL); // store the area length within the flux
        };

        updatePredictor(i, j, IX);
        updatePredictor(i, j, IY);
      });

    // corrector
    Kokkos::parallel_for(
      "Update CTU corrector", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        State un_loc = getStateFromArray(Unew, i, j);

        // Calling the right Riemann solver
        auto riemann = [&](State qL, State qR, State &flux, Pos &rot, real_t &pout) {
          qL = rotate(qL , rot);
          qR = rotate(qR , rot);
          switch (params.riemann_solver) {
            case HLL: hll(qL, qR, flux, pout, params); break;
            default: hllc(qL, qR, flux, pout, params); break;
          }
          flux = rotate_back(flux, rot);
        };

        // Lambda to update the cell along a direction
        auto updateCorrector = [&](int i, int j, IDir dir) {

          auto reconstructTransverse = [&](ISide side, Pos &normal) -> Kokkos::Array<State, 2> {
            const IDir tdir = (dir == IX ? IY : IX); // 2d case
            const int  dxp  = (dir == IX ?  1 :  0);
            const int  dyp  = (dir == IY ?  1 :  0);
            auto [qL, qR] = reconstruct(Q, slopesX, slopesY, psi, i, j, dir, side,  geometry, params);
            Pos tangential = {normal[IY], -normal[IX]};
            
            auto applyTransverseFluxes = [&](State &U, ISide side2){
              int ii = i + (side == ILEFT ? -1 : 1) * (side == side2  && dir == IX);
              int jj = j + (side == ILEFT ? -1 : 1) * (side == side2  && dir == IY);

              real_t lenTL, lenTR;
              {
                Pos normalTL = geometry.getRotationMatrix(ii, jj, tdir, ILEFT,  lenTL);
                Pos normalTR = geometry.getRotationMatrix(ii, jj, tdir, IRIGHT, lenTR);
                lenTL = lenTL * fabs(dot(normalTL, tangential));
                lenTR = lenTR * fabs(dot(normalTR, tangential));
              }
              const real_t cellAreaT = geometry.cellArea(ii, jj);

              State FL = getStateFromArray(Fhat[tdir], ii,     jj);
              State FR = getStateFromArray(Fhat[tdir], ii+dyp, jj+dxp); // flux direction transverse
              U  = primToCons(U, params);
              U = U + 0.5 * dt * (lenTL*FL - lenTR*FR) / cellAreaT;
              U = consToPrim(U, params);
            };

            applyTransverseFluxes(qL, ILEFT);
            applyTransverseFluxes(qR, IRIGHT);
            return {qL, qR};
          };

          real_t lenL, lenR;
          Pos rotL = geometry.getRotationMatrix(i, j, dir, ILEFT,  lenL);
          Pos rotR = geometry.getRotationMatrix(i, j, dir, IRIGHT, lenR);
          real_t cellArea = geometry.cellArea(i,j);

          auto [qL, qCL] = reconstructTransverse(ILEFT, rotL);
          auto [qCR, qR] = reconstructTransverse(IRIGHT, rotR);

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, rotL, poutL);
          riemann(qCR, qR, fluxR, rotR, poutR);

          un_loc += dt * (lenL*fluxL - lenR*fluxR) / cellArea;
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


    auto dparams = full_params.device_params;
    Geometry geometry = this->geometry;
    Kokkos::parallel_for(
      "alpha_beta conversion",
      full_params.range_tot,
      KOKKOS_LAMBDA(const int i, const int j) {
        const real_t r = norm(geometry.mapc2p_center(i,j));
        const real_t rho0 = dparams.spl_rho(r);
        const real_t prs0 = dparams.spl_prs(r);
        
        Q(j, i, IR) = Q(j, i, IR) / rho0;
        Q(j, i, IP) = Q(j, i, IP) / prs0;
      });

    
    computeSlopes(Q);
    if (full_params.device_params.reconstruction == RECONS_BJ)
      computeLimiterBJ(Q);
    if (full_params.hancock_ts)
      computeHancockStep(Q, dt);

    // Hyperbolic udpate
    switch(full_params.flux_solver) { 
      case FLUX_GODOUNOV: computeFluxesAndUpdate(Q, Unew, dt); break;
      case FLUX_CTU:      computeFluxesAndUpdate_CTU(Q, Unew, dt); break;
    }

    // Splitted terms
    if (full_params.device_params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (full_params.device_params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
    // if (full_params.device_params.gravity != GRAV_NONE)
    //   grav_functor.applyGravity(Q, Unew, dt);
    if (full_params.device_params.coriolis_active)
      coriolis_functor.applyCoriolis(Q, Unew, dt);
    if (full_params.device_params.heating != HEAT_NONE)
      heating_functor.applyHeating(Q, Unew, dt);
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

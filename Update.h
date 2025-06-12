#pragma once 

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"
#include "Gravity.h"

namespace fv2d {

using WENOArray = Kokkos::View<real_t****>;

namespace {
  struct WENOStruct {
    WENOArray PxL, PxR, PyL, PyR, BetaX, BetaY, WeightXL, WeightXR, WeightYL, WeightYR;
  };

  /**
   * @brief Reconstructs interfqce stqte using WENO3 method
   * 
   * @param Q The primitive variables array
   */
  KOKKOS_INLINE_FUNCTION
  State reconstruct_weno3(Array Q, IDir dir, real_t sign, WENOStruct weno_struct, int i, int j, const DeviceParams &params) {
    State res;
    
    // .... 
    auto &PxL = weno_struct.PxL;
    auto &PxR = weno_struct.PxR;
    auto &PyL = weno_struct.PyL;
    auto &PyR = weno_struct.PyR;
    auto &BetaX = weno_struct.BetaX;
    auto &BetaY = weno_struct.BetaY;
    auto &WeightXL = weno_struct.WeightXL;
    auto &WeightXR = weno_struct.WeightXR;
    auto &WeightYL = weno_struct.WeightYL;
    auto &WeightYR = weno_struct.WeightYR;
    

    if (dir == IX){
      if (sign <0){
        for (int ivar=0; ivar < Nfields; ++ivar){
          res[ivar] = WeightXL(0, j, i, ivar) * PxL(0, j, i, ivar) + WeightXL(1, j, i, ivar) * PxL(1, j, i, ivar) + WeightXL(2, j, i, ivar) * PxL(2, j, i, ivar);
        };
        /*
        if (i==19 && j==8){
          printf("rho R = %lf; centered = %lf\n", res[IR], Q(j, i, IR));
          printf("BetaX 0 = %lf\n", BetaX(0, j, i, IR));
          printf("BetaX 2 = %lf\n", BetaX(2, j, i, IR));
          printf("weight 0 = %lf\n", WeightXL(0, j, i, IR));
          printf("weight 2 = %lf\n", WeightXL(2, j, i, IR));
        }*/
        
      } else {
        for (int ivar=0; ivar < Nfields; ++ivar){
          res[ivar] = WeightXR(0, j, i, ivar) * PxR(0, j, i, ivar) + WeightXR(1, j, i, ivar) * PxR(1, j, i, ivar) + WeightXR(2, j, i, ivar) * PxR(2, j, i, ivar);
        };
        /*
        if (i==18 && j==8){
          printf("rho L = %lf; centered = %lf\n", res[IR], Q(j, i, IR));
          printf("BetaX 0 = %lf\n", BetaX(0, j, i, IR));
          printf("BetaX 2 = %lf\n", BetaX(2, j, i, IR));
          printf("weight 0 = %lf\n", WeightXL(0, j, i, IR));
          printf("weight 2 = %lf\n", WeightXR(2, j, i, IR));
        }*/
      }
    } else if (dir == IY){
      if (sign <0){
        for (int ivar=0; ivar < Nfields; ++ivar){
          res[ivar] = WeightYL(0, j, i, ivar) * PyL(0, j, i, ivar) + WeightYL(1, j, i, ivar) * PyL(1, j, i, ivar) + WeightYL(2, j, i, ivar) * PyL(2, j, i, ivar);
        }
      } else {
        for (int ivar=0; ivar < Nfields; ++ivar){
          res[ivar] = WeightYR(0, j, i, ivar) * PyR(0, j, i, ivar) + WeightYR(1, j, i, ivar) * PyR(1, j, i, ivar) + WeightYR(2, j, i, ivar) * PyR(2, j, i, ivar);
        }
      }
    }
    

    return res;
  }

  /**
   * @brief MUSCL-type reconstruction for the hydro update
   * 
   * Options are :
   *  . PCM : Piecewise Constant Method, taking the original value of the cell (1st order)
   *  . PLM : Piecewise Linear Method, using the slopes to reconstruct (2nd order)
   *  . PCM_WB : Piecewise Constant Method + Well-balancing (1st order), taken from Kappeli & Mishra 2016
   *  . PLM_WB : Piecewise Linear Method + Well-balancing (2nd order), taken from Kappeli & Mishra 2016
   *  . WENO3 : Weighted Essentially Non Oscillatory Scheme 
   */
  KOKKOS_INLINE_FUNCTION
  State reconstruct(Array Q, Array slopes, WENOStruct weno_struct, int i, int j, real_t sign, IDir dir, const DeviceParams &params) {
    State q     = getStateFromArray(Q, i, j);
    
    State res;
    switch (params.reconstruction) {
      case PLM: 
      {
        State slope = getStateFromArray(slopes, i, j);
        res = q + slope * sign * 0.5; break; // Piecewise Linear
      } 
      case PCM_WB: // Piecewise constant + Well-balancing
      {
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IP] = q[IP] + sign * q[IR] * getGravity(i, j, dir, params) * params.dy * 0.5;
      }
      break;
      case PLM_WB: // Piecewise linear + Well-balancing 
      {
        // Piecewise linear reconstruction
        /*res = q + slope * sign * 0.5;

        if (dir == IY) {
          // Getting neighbour states
          State neigh_m, neigh_p;
          neigh_m = getStateFromArray(Q, i + (dir == IX ? -1 : 0), j + (dir == IY ? -1 : 0));
          neigh_p = getStateFromArray(Q, i + (dir == IX ?  1 : 0), j + (dir == IY ?  1 : 0));
          
          // Calculating p1_{i-1}
          const real_t p0_im = q[IP] + 0.5 * (q[IR] + neigh_m[IR]) * params.g * params.dy;
          const real_t p1_im = neigh_m[IP] - p0_im;

          // Calculating p1_{i+1}
          const real_t p0_ip = q[IP] - 0.5 * (q[IR] + neigh_p[IR]) * params.g * params.dy;
          const real_t p1_ip = neigh_p[IP] - p0_ip;

          // TODO : This should be outsourced to compute Slopes !!!
          // Calculating slope using minmod
          real_t dP = 0.0;
          if (p1_im * p1_ip < 0.0) // Slope left is -p1_im; Slope right is p1_ip so their sign should be opposite
            dP = (Kokkos::abs(p1_im) < Kokkos::abs(p1_ip) ? -p1_im : p1_ip);

          // Reconstructing pressure
          res[IP] = (sign == -1 ? p0_im : p0_ip) + dP * sign * 0.5;
        }*/

      } break;
      case WENO3: res = reconstruct_weno3(Q, dir, sign, weno_struct, i, j, params); break;
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

  // Muscl Slopes
  Array slopesX, slopesY;

  // WENO3
  WENOArray PxL, PxR, PyL, PyR, BetaX, BetaY, WeightXL, WeightXR, WeightYL, WeightYR;

  UpdateFunctor(const Params &full_params)
    : full_params(full_params), bc_manager(full_params),
      tc_functor(full_params), visc_functor(full_params) {
      auto device_params = full_params.device_params;
      if (device_params.reconstruction == PLM) {
        slopesX = Array("SlopesX", device_params.Nty, device_params.Ntx, Nfields);
        slopesY = Array("SlopesY", device_params.Nty, device_params.Ntx, Nfields);
      }
      else if (device_params.reconstruction == WENO3) {
        PxL = WENOArray("PxL", 3, device_params.Nty, device_params.Ntx, Nfields);
        PxR = WENOArray("PxR", 3, device_params.Nty, device_params.Ntx, Nfields);
        PyL = WENOArray("PyL", 3, device_params.Nty, device_params.Ntx, Nfields);
        PyR = WENOArray("Pyr", 3, device_params.Nty, device_params.Ntx, Nfields);
        BetaX  = WENOArray("BetaX",  3, device_params.Nty, device_params.Ntx, Nfields);
        BetaY  = WENOArray("BetaY",  3, device_params.Nty, device_params.Ntx, Nfields);
        WeightXL  = WENOArray("WeightXL",  3, device_params.Nty, device_params.Ntx, Nfields);
        WeightXR  = WENOArray("WeightXR",  3, device_params.Nty, device_params.Ntx, Nfields);
        WeightYL  = WENOArray("WeightYL",  3, device_params.Nty, device_params.Ntx, Nfields);
        WeightYR  = WENOArray("WeightYR",  3, device_params.Nty, device_params.Ntx, Nfields);
      }
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

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = full_params.device_params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;

    WENOStruct weno_struct {PxL, PxR, PyL, PyR, BetaX, BetaY, WeightXL, WeightXR, WeightYL, WeightYR};

    Kokkos::parallel_for(
      "Update", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : slopesY);
          int dxm = (dir == IX ? -1 : 0);
          int dxp = (dir == IX ?  1 : 0);
          int dym = (dir == IY ? -1 : 0);
          int dyp = (dir == IY ?  1 : 0);

          State qCR = reconstruct(Q, slopes, weno_struct, i, j,  1.0, dir, params);
          State qCL = reconstruct(Q, slopes, weno_struct, i, j, -1.0, dir, params);
          State qL  = reconstruct(Q, slopes, weno_struct, i+dxm, j+dym, 1.0, dir, params);
          State qR  = reconstruct(Q, slopes, weno_struct, i+dxp, j+dyp, -1.0, dir, params);

          const real_t gdx = (dir == IX ? 0.0 : params.g * params.dy);

          // Calling the right Riemann solver
          auto riemann = [&](State qL, State qR, State &flux, int side, real_t &pout) {
            switch (params.riemann_solver) {
              case HLL:  hll(qL, qR, flux, pout, params); break;
              case FSLP: fslp(qL, qR, flux, pout, side, gdx, params); break;
              default:   hllc(qL, qR, flux, pout, params); break;
            }
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, -1, poutL);
          riemann(qCR, qR, fluxR,  1, poutR);

          fluxL = swap_component(fluxL, dir);
          fluxR = swap_component(fluxR, dir);

          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_y_bc && (j==params.jbeg || j==params.jend-1) && dir == IY) {
            real_t g = getGravity(i, j, dir, params);
            if (j==params.jbeg)
              fluxL = State{0.0, 0.0, poutR - Q(j, i, IR)*g*params.dy, 0.0};
            else 
              fluxR = State{0.0, 0.0, poutL + Q(j, i, IR)*g*params.dy, 0.0};
          }

          auto un_loc = getStateFromArray(Unew, i, j);
          un_loc += dt*(fluxL - fluxR)/(dir == IX ? params.dx : params.dy);
        
          if (params.gravity_mode != GRAV_NONE && params.riemann_solver != FSLP) {
            real_t g = getGravity(i, j, dir, params);
            un_loc[IV] += dt * Q(j, i, IR) * g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * g;
          }

          setStateInArray(Unew, i, j, un_loc);
        };

        updateAlongDir(i, j, IX);
        updateAlongDir(i, j, IY);
      });
  }

  void compute_weno_beta(Array Q) {
    auto BetaX = this->BetaX;
    auto BetaY = this->BetaY;

    Kokkos::parallel_for(
      "Weno Beta",
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        for (int ivar=0; ivar < Nfields; ++ivar) {
          ////// Selon X
          BetaX(0, j, i, ivar) = 13/12.0 * (Q(j, i-2, ivar) - 2*Q(j, i-1, ivar) + Q(j, i, ivar)) * (Q(j, i-2, ivar) - 2*Q(j, i-1, ivar) + Q(j, i, ivar))
                              + 1/4.0 * (Q(j, i-2, ivar) - 4*Q(j, i-1, ivar) + 3*Q(j, i, ivar)) * (Q(j, i-2, ivar) - 4*Q(j, i-1, ivar) + 3*Q(j, i, ivar));
          BetaX(1, j, i, ivar) = 13/12.0 * (Q(j, i-1, ivar) - 2*Q(j, i, ivar) + Q(j, i+1, ivar)) * (Q(j, i-1, ivar) - 2*Q(j, i, ivar) + Q(j, i+1, ivar))
                              + 1/4.0 * (Q(j, i-1, ivar) - Q(j, i+1, ivar)) * (Q(j, i-1, ivar) - Q(j, i+1, ivar));
          BetaX(2, j, i, ivar) = 13/12.0 * (Q(j, i, ivar) - 2*Q(j, i+1, ivar) + Q(j, i+2, ivar)) * (Q(j, i, ivar) - 2*Q(j, i+1, ivar) + Q(j, i+2, ivar))
                              + 1/4.0 * (3*Q(j, i, ivar) - 4*Q(j, i+1, ivar) + Q(j, i+2, ivar)) * (3*Q(j, i, ivar) - 4*Q(j, i+1, ivar) + Q(j, i+2, ivar));


          
          ////// Selon Y
          BetaY(0, j, i, ivar) = 13/12.0 * (Q(j-2, i, ivar) - 2*Q(j-1, i, ivar) + Q(j, i, ivar)) * (Q(j-2, i, ivar) - 2*Q(j-1, i, ivar) + Q(j, i, ivar))
                              + 1/4.0 * (Q(j-2, i, ivar) - 4*Q(j-1, i, ivar) + 3*Q(j, i, ivar)) * (Q(j-2, i, ivar) - 4*Q(j-1, i, ivar) + 3*Q(j, i, ivar));
          BetaY(1, j, i, ivar) = 13/12.0 * (Q(j-1, i, ivar) - 2*Q(j, i, ivar) + Q(j+1, i, ivar)) * (Q(j-1, i, ivar) - 2*Q(j, i, ivar) + Q(j+1, i, ivar))
                              + 1/4.0 * (Q(j-1, i, ivar) - Q(j+1, i, ivar)) * (Q(j-1, i, ivar) - Q(j+1, i, ivar));
          BetaY(2, j, i, ivar) = 13/12.0 * (Q(j, i, ivar) - 2*Q(j+1, i, ivar) + Q(j+2, i, ivar)) * (Q(j, i, ivar) - 2*Q(j+1, i, ivar) + Q(j+2, i, ivar))
                              + 1/4.0 * (3*Q(j, i, ivar) - 4*Q(j+1, i, ivar) + Q(j+2, i, ivar)) * (3*Q(j, i, ivar) - 4*Q(j+1, i, ivar) + Q(j+2, i, ivar));

        }
      });
  }
  
  void compute_weno_w(Array Q) {
    auto WeightXL = this->WeightXL;
    auto WeightXR = this->WeightXR;
    auto WeightYL = this->WeightYL;
    auto WeightYR = this->WeightYR;
    auto BetaX = this->BetaX;
    auto BetaY = this->BetaY;

    Kokkos::parallel_for(
      "Weno Weights",
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {

      real_t eps = 1.0e-6; //epsilon choice

        for (int ivar=0; ivar < Nfields; ++ivar) {
          ////// Selon X
          // Right
          WeightXR(0, j, i, ivar) = 1/10.0 * 1/((eps + BetaX(0, j, i, ivar)) * (eps + BetaX(0, j, i, ivar)));
          WeightXR(1, j, i, ivar) = 3/5.0 * 1/((eps + BetaX(1, j, i, ivar)) * (eps + BetaX(1, j, i, ivar)));
          WeightXR(2, j, i, ivar) = 3/10.0 * 1/((eps + BetaX(2, j, i, ivar)) * (eps + BetaX(2, j, i, ivar)));

          real_t sumXR = WeightXR(0, j, i, ivar) + WeightXR(1, j, i, ivar) + WeightXR(2, j, i, ivar);

          WeightXR(0, j, i, ivar) /= sumXR;
          WeightXR(1, j, i, ivar) /= sumXR;
          WeightXR(2, j, i, ivar) /= sumXR;

          // Left
          WeightXL(0, j, i, ivar) = 3/10.0 * 1/((eps + BetaX(0, j, i, ivar)) * (eps + BetaX(0, j, i, ivar)));
          WeightXL(1, j, i, ivar) = 3/5.0 * 1/((eps + BetaX(1, j, i, ivar)) * (eps + BetaX(1, j, i, ivar)));
          WeightXL(2, j, i, ivar) = 1/10.0 * 1/((eps + BetaX(2, j, i, ivar)) * (eps + BetaX(2, j, i, ivar)));

          real_t sumXL = WeightXL(0, j, i, ivar) + WeightXL(1, j, i, ivar) + WeightXL(2, j, i, ivar);

          WeightXL(0, j, i, ivar) /= sumXL;
          WeightXL(1, j, i, ivar) /= sumXL;
          WeightXL(2, j, i, ivar) /= sumXL;


          ////// Selon Y
          // Right
          WeightYR(0, j, i, ivar) = 1/10.0 * 1/((eps + BetaY(0, j, i, ivar)) * (eps + BetaY(0, j, i, ivar)));
          WeightYR(1, j, i, ivar) = 3/5.0 * 1/((eps + BetaY(1, j, i, ivar)) * (eps + BetaY(1, j, i, ivar)));
          WeightYR(2, j, i, ivar) = 3/10.0 * 1/((eps + BetaY(2, j, i, ivar)) * (eps + BetaY(2, j, i, ivar)));

          real_t sumYR = WeightYR(0, j, i, ivar) + WeightYR(1, j, i, ivar) + WeightYR(2, j, i, ivar);

          WeightYR(0, j, i, ivar) /= sumYR;
          WeightYR(1, j, i, ivar) /= sumYR;
          WeightYR(2, j, i, ivar) /= sumYR;
          
          // Left
          WeightYL(0, j, i, ivar) = 3/10.0 * 1/((eps + BetaY(0, j, i, ivar)) * (eps + BetaY(0, j, i, ivar)));
          WeightYL(1, j, i, ivar) = 3/5.0 * 1/((eps + BetaY(1, j, i, ivar)) * (eps + BetaY(1, j, i, ivar)));
          WeightYL(2, j, i, ivar) = 1/10.0 * 1/((eps + BetaY(2, j, i, ivar)) * (eps + BetaY(2, j, i, ivar)));

          real_t sumYL = WeightYL(0, j, i, ivar) + WeightYL(1, j, i, ivar) + WeightYL(2, j, i, ivar);

          WeightYL(0, j, i, ivar) /= sumYL;
          WeightYL(1, j, i, ivar) /= sumYL;
          WeightYL(2, j, i, ivar) /= sumYL;

        }
      });

  }

  void compute_weno_P(Array Q) {
    auto PxL = this->PxL;
    auto PxR = this->PxR;
    auto PyL = this->PyL;
    auto PyR = this->PyR;

    Kokkos::parallel_for(
      "Weno Beta",
      full_params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j) {
        for (int ivar=0; ivar < Nfields; ++ivar) {
          ////// Selon X
          PxR(0, j, i, ivar) = 1/3.0 * Q(j, i-2, ivar) - 7/6.0 * Q(j, i-1, ivar) + 11/6.0 * Q(j, i, ivar);
          PxL(0, j, i, ivar) = -1/6.0 * Q(j, i-2, ivar) + 5/6.0 * Q(j, i-1, ivar) + 1/3.0 * Q(j, i, ivar);

          PxR(1, j, i, ivar) = -1/6.0 * Q(j, i-1, ivar) + 5/6.0 * Q(j, i, ivar) + 1/3.0 * Q(j, i+1, ivar);
          PxL(1, j, i, ivar) = 1/3.0 * Q(j, i-1, ivar) + 5/6.0 * Q(j, i, ivar) - 1/6.0 * Q(j, i+1, ivar);

          PxR(2, j, i, ivar) = 1/3.0 * Q(j, i, ivar) + 5/6.0 * Q(j, i+1, ivar) - 1/6.0 * Q(j, i+2, ivar);
          PxL(2, j, i, ivar) = 11/6.0 * Q(j, i, ivar) - 7/6.0 * Q(j, i+1, ivar) + 1/3.0 * Q(j, i+2, ivar);

          ////// Selon Y
          PyR(0, j, i, ivar) = 1/3.0 * Q(j-2, i, ivar) - 7/6.0 * Q(j-1, i, ivar) + 11/6.0 * Q(j, i, ivar);
          PyL(0, j, i, ivar) = -1/6.0 * Q(j-2, i, ivar) + 5/6.0 * Q(j-1, i, ivar) + 1/3.0 * Q(j, i, ivar);

          PyR(1, j, i, ivar) = -1/6.0 * Q(j-1, i, ivar) + 5/6.0 * Q(j, i, ivar) + 1/3.0 * Q(j+1, i, ivar);
          PyL(1, j, i, ivar) = 1/3.0 * Q(j-1, i, ivar) + 5/6.0 * Q(j, i, ivar) - 1/6.0 * Q(j+1, i, ivar);

          PyR(2, j, i, ivar) = 1/3.0 * Q(j, i, ivar) + 5/6.0 * Q(j+1, i, ivar) - 1/6.0 * Q(j+2, i, ivar);
          PyL(2, j, i, ivar) = 11/6.0 * Q(j, i, ivar) - 7/6.0 * Q(j+1, i, ivar) + 1/3.0 * Q(j+2, i, ivar);

        }
      });

  }

  void euler_step(Array Q, Array Unew, real_t dt) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hyperbolic udpate
    if (full_params.device_params.reconstruction == PLM)
      computeSlopes(Q);
    else if (full_params.device_params.reconstruction == WENO3) {
      compute_weno_beta(Q);
      compute_weno_w(Q);
      compute_weno_P(Q);
    }
    computeFluxesAndUpdate(Q, Unew, dt);

    // Splitted terms
    if (full_params.device_params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (full_params.device_params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
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
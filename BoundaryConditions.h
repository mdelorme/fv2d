#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"
#include "Gravity.h"

namespace fv2d {
  namespace {
  /**
   * @brief Absorbing conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillAbsorbing(Array Q, int iref, int jref) {
    return getStateFromArray(Q, iref, jref);
  };

  /**
   * @brief Reflecting boundary conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillReflecting(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params) {
    int isym, jsym;
    if (dir == IX) {
      int ipiv = (i < iref ? params.ibeg : params.iend);
      isym = 2*ipiv - i - 1;
      jsym = j;
    }
    else {
      int jpiv = (j < jref ? params.jbeg : params.jend);
      isym = i;
      jsym = 2*jpiv - j - 1;
    }

    State q = getStateFromArray(Q, isym, jsym);

    if (dir == IX) {
      q[IU] *= -1.0;
      #ifdef MHD
      q[IBX] *= -1.0;
      #endif
    }
    else {
      q[IV] *= -1.0;
      #ifdef MHD
      q[IBY] *= -1.0;
      #endif
    }
    return q;
  }

  /**
   * @brief Periodic boundary conditions
   *
   */
  KOKKOS_INLINE_FUNCTION
  State fillPeriodic(Array Q, int i, int j, IDir dir, const DeviceParams &params) {
    if (dir == IX) {
      if (i < params.ibeg)
        i += params.Nx;
      else
        i -= params.Nx;
    }
    else {
      if (j < params.jbeg)
        j += params.Ny;
      else
        j -= params.Ny;
    }

    return getStateFromArray(Q, i, j);
  }

  /**
   * @brief Experimental stuff for tri-layer
   */
  KOKKOS_INLINE_FUNCTION
  State fillTriLayerDamping(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params) {
    if (dir == IY && j < 0) {
      Pos pos = getPos(params, i, j);
      const real_t T0 = params.iso3_T0;
      const real_t rho0 = params.iso3_rho0 * exp(-params.iso3_dy0 * params.gy / T0);
      const real_t p0 = rho0*T0;
      const real_t d = pos[IY];

      // Top layer (iso-thermal)
      real_t rho, p;
      p   = p0 * exp(params.gy * d / T0);
      rho = p / T0;

      State q;

      q = getStateFromArray(Q, i, jref);
      q[IR] = rho;
      // q[IU] = 0.0;
      // q[IV] = 0.0;
      q[IP] = p;
      return q;
    }
    else
      return fillAbsorbing(Q, iref, jref);
  }



  KOKKOS_INLINE_FUNCTION
  void applyWellBalanced(Array Q, int i, int j, State &flux_tot, const real_t pout, const DeviceParams &params) {
    real_t g = getGravity(i, j, IY, params);
    real_t sign = (j==params.jbeg ? 1.0 : -1.0);
    // We overwite the hydro flux
    // But what to do with the magnetic part?
    flux_tot[IR] = 0.0;
    flux_tot[IU] = 0.0;
    flux_tot[IV] = pout + sign * Q(j, i, IR) * g * params.dy;
    flux_tot[IW] = 0.0;
    flux_tot[IE] = 0.0;
  }

  #ifdef MHD

    /**
   * @brief For the tri-layer test case with magnetic field, we want a normal field BC, and also a WB scheme, the rest
   * of the variables are copied from the domain, we have absorbing boundaries.
   */
  KOKKOS_INLINE_FUNCTION
  State fillTriLayerMag(Array Q, int i, int j, IDir dir, const DeviceParams &params) {
    State q = getStateFromArray(Q, i, j);
    if ((j==params.jbeg || j==params.jend-1) && dir == IY) {
      q[IV]  = 0.0; // Vitesse normal nulle, pas de matière qui entre ou sort
      q[IBX] = 0.0;
      q[IBZ] = 0.0; // Champ magnétique normal -> Bx=Bz=0 et By_j+1 = By_j
    }
    return q;
  }

  /**
   * @brief Boundary values and flux for a normal Magnetic Field Boundary Condition.
   * For normal fields, Bx=Bz=0 at the boundary. 
   */
  KOKKOS_INLINE_FUNCTION
  void applyNormalMagBC(Array Q, int i, int j, State &flux_tot, IDir dir, const DeviceParams &params) {
    State q = getStateFromArray(Q, i, j);
    Vect v {q[IU], q[IV], q[IW]};
    Vect B {q[IBX], q[IBY], q[IBZ]};
    if (params.well_balanced_flux_at_y_bc)
      v[IY] = 0.0; // Vitesse normale nulle, pas de matière qui entre ou sort
    // IDir IBN = (dir == IX ? IBX : IBY); // Index of the normal component of B
    // B[dir] = q[IBN];
    State flux_mhd = zero_state();
    if (dir == IY){
      flux_mhd[IV]  = -0.5 * B[IY]*B[IY];
      flux_mhd[IBX] = -v[IX] * B[IY];
      flux_mhd[IBZ] = -v[IZ] * B[IY];
    }
    if (params.riemann_solver == IDEALGLM || params.div_cleaning == DEDNER) {
      State qL, qR;
      if (j==params.jbeg) {
        qL = getStateFromArray(Q, i, j);
        qR = q;
      }
      else {
        qR = getStateFromArray(Q, i, j);
        qL = q;
      }
      real_t Bm = qL[IBY]  + 0.5 * (qR[IBY] - qL[IBY]) - 1/(2*c_h) * (qR[IPSI] - qL[IPSI]);
      real_t psi_m = qL[IPSI] + 0.5 * (qR[IPSI] - qL[IPSI]) - 0.5*c_h * (qR[IBY] - qL[IBY]);
      // flux_mhd[IE] += c_h * q[IPSI] * B[dir];

      flux_mhd[IBY] = psi_m;
      flux_mhd[IPSI] = c_h * c_h * Bm;
    }
    flux_tot = flux_tot + flux_mhd;
  }


  KOKKOS_INLINE_FUNCTION
  State applyTriLayersBoundaries(Array Q, int i, int j, IDir dir, const real_t poutL, const real_t poutR, const real_t c_h, const DeviceParams &params) {
    // TODO: Prendre en compte les autres types de BC si différent de absorbant
    // TODO: Plus logique surement de passer les flux et de les modifier "inplace".
    State q = getStateFromArray(Q, i, j);
    const Vect v {q[IU], 0.0, q[IW]};
    const Vect B {0.0, q[IBY], 0.0};

    State flux_hydro = zero_state();
    if (params.well_balanced_flux_at_y_bc &&  (j==params.jbeg || j==params.jend-1) && dir == IY) {
      real_t g = getGravity(i, j, dir, params);
      if (j==params.jbeg){
        flux_hydro = zero_state();
        flux_hydro[IV] = poutR - q[IR]*g*params.dy;
      }
      else{
        flux_hydro = zero_state();
        flux_hydro[IV] = poutL + q[IR]*g*params.dy;
      }
    }
    
    State flux_mhd = zero_state();
    flux_mhd[IV]  = -0.5 * B[IY]*B[IY];
    flux_mhd[IBX] = -v[IX] * B[IY];
    flux_mhd[IBZ] = -v[IZ] * B[IY];

    if (params.riemann_solver == IDEALGLM || params.div_cleaning == DEDNER) {
      State qL, qR;
      if (j==params.jbeg) {
        qL = getStateFromArray(Q, i, j);
        qR = q;
      }
      else {
        qR = getStateFromArray(Q, i, j);
        qL = q;
      }
      real_t Bm = qL[IBY]  + 0.5 * (qR[IBY] - qL[IBY]) - 1/(2*c_h) * (qR[IPSI] - qL[IPSI]);
      real_t psi_m = qL[IPSI] + 0.5 * (qR[IPSI] - qL[IPSI]) - 0.5*c_h * (qR[IBY] - qL[IBY]);
      // flux_mhd[IE] += c_h * q[IPSI] * B[dir];

      flux_mhd[IBY] = psi_m;
      flux_mhd[IPSI] = c_h * c_h * Bm;
    }
    return flux_hydro + flux_mhd;
  }
  # endif //MHD
} // anonymous namespace


class BoundaryManager {
public:
  Params full_params;

  BoundaryManager(const Params &full_params) 
    : full_params(full_params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q) {
    auto params = full_params.device_params;
    auto bc_x = params.boundary_x;
    auto bc_y = params.boundary_y;

    Kokkos::parallel_for( "Filling X-boundary",
                          full_params.range_xbound,
                          KOKKOS_LAMBDA(int i, int j) {

                            int ileft     = i;
                            int iright    = params.iend+i;
                            int iref_left = params.ibeg;
                            int iref_right = params.iend-1;

                            auto fill = [&](int i, int iref) {
                              switch (bc_x) {
                                case BC_ABSORBING:  return fillAbsorbing(Q, iref, j); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, iref, j, IX, params); break;
                                default:   return fillPeriodic(Q, i, j, IX, params); break;
                              }
                            };

                            setStateInArray(Q, ileft, j,  fill(ileft, iref_left));
                            setStateInArray(Q, iright, j, fill(iright, iref_right));
                          });

    Kokkos::parallel_for( "Filling Y-boundary",
                          full_params.range_ybound,
                          KOKKOS_LAMBDA(int i, int j) {

                            int jtop     = j;
                            int jbot     = params.jend+j;
                            int jref_top = params.jbeg;
                            int jref_bot = params.jend-1;

                            auto fill = [&](int j, int jref) {
                              switch (bc_y) {
                                case BC_ABSORBING:  return fillAbsorbing(Q, i, jref); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, i, jref, IY, params); break;
                                case BC_TRILAYER_DAMPING: return fillTriLayerDamping(Q, i, j, i, jref, IY, params); break;
                                case BC_MAG_TRILAYER: return fillTriLayerMag(Q, i, jref, IY, params); break;
                                default:   return fillPeriodic(Q, i, j, IY, params); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}

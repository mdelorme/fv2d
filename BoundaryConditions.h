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

  #ifdef MHD
  /**
   * @brief Boundary values and flux for a normal Magnetic Field Boundary Condition.
   * For normal fields, Bx=Bz=0 at the boundary. 
   */
  KOKKOS_INLINE_FUNCTION
  State getNormalMagFieldFlux(State &u, int i, int j, IDir dir, const real_t poutL, const real_t poutR, const real_t c_h, const DeviceParams &params) {
    // TODO: Prendre en compte les autres types de BC si différent de absorbant
    // TODO: Plus logique surement de passer les flux et de les modifier "inplace".
    State q = consToPrim(u, params);
    q[IBX] = 0.0;
    q[IBZ] = 0.0;
    const Vect v {q[IU], q[IV], q[IW]};
    const Vect B {q[IBX], q[IBY], q[IBZ]};
    const real_t vnormal = v[dir];
    const real_t e_hydro = u[IE] - 0.5 * norm2(B);

    State flux_hydro = zero_state();
    flux_hydro[IR] = q[IR] * vnormal;
    flux_hydro[IU] = q[IR] * vnormal * v[IX];
    flux_hydro[IV] = q[IR] * vnormal * v[IY];
    flux_hydro[IW] = q[IR] * vnormal * v[IZ];
    flux_hydro[IE] = (e_hydro + q[IP]) * vnormal;
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
    const real_t Bnormal = B[dir];
    Vect ptot {0.0, 0.0, 0.0};
    ptot[dir] = getMagneticPressure(B); //q[IP] + 
    
    State flux_mhd = zero_state();
    flux_mhd[IU]  = -B[IX] * Bnormal + ptot[IX];
    flux_mhd[IV]  = -B[IY] * Bnormal + ptot[IY];
    flux_mhd[IW]  = -B[IZ] * Bnormal + ptot[IZ];
    flux_mhd[IE]  =  norm2(B) * vnormal - Bnormal * dot(v, B);  // We add +0.5B^2 for the energy contribution and, 0.5*B^2 for the pressure contribution (see eq. 3.21 Derigs et al. 2018)
    flux_mhd[IBX] =  B[IX] * vnormal - v[IX] * Bnormal;
    flux_mhd[IBY] =  B[IY] * vnormal - v[IY] * Bnormal;
    flux_mhd[IBZ] =  B[IZ] * vnormal - v[IZ] * Bnormal;

    // if (params.riemann_solver == IDEALGLM || params.div_cleaning == DEDNER) {
    //   IVar IBN = (dir == IX ? IBX : IBY);
    //   flux_mhd[IE] += c_h * q[IPSI] * Bnormal; 
    //   flux_mhd[IBN] = c_h * q[IPSI];
    //   flux_mhd[IPSI] = c_h * B[dir];
    // }
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
                                default:   return fillPeriodic(Q, i, j, IY, params); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}

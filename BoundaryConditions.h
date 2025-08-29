#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"

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
   * @brief Boundary values and flux for a normal Magnetic Field Boundary Condition
   */
  KOKKOS_INLINE_FUNCTION
  State getNormalMagFieldAndFlux(State u, State &flux, int i, int j, real_t ch, IDir dir, const DeviceParams &params) {
    // Assume we're at the y_min or y_max boundary
    if (dir == IX) {
      return u;
    }
    
    State q = consToPrim(u, params);
    q[IBX] = 0.0;
    // q[IBY] = (j==params.jbeg) ? params.bcmag_ymax_value : params.bcmag_ymin_value;
    q[IBZ] = 0.0;
    if (params.riemann_solver == IDEALGLM || params.div_cleaning == DEDNER)
      q[IPSI] = 0.0;
    // We should also recompute the flux
    // Hydro Flux
    flux[IR] = q[IR] * q[IV];
    flux[IU] = flux[IR] * q[IU];
    flux[IV] = flux[IR] * q[IV] + q[IP];
    flux[IW] = flux[IR] * q[IW];
    flux[IE] = (q[IP] + u[IE]) * q[IV]; // Note, u[IE] already contains the mag energy.
    // Magnetic Flux
    const real_t udotB = q[IBX]*q[IU] + q[IBY]*q[IV] + q[IBZ]*q[IW];
    flux[IBX] = q[IBX] * q[IV] - q[IBY] * q[IU];
    flux[IBY] = q[IBY] * q[IV] - q[IBY] * q[IV];
    flux[IBZ] = q[IBZ] * q[IV] - q[IBY] * q[IW];

    // Modification of the hydro flux
    const real_t pmag = getMagneticPressure({q[IBX], q[IBY], q[IBZ]});
    flux[IU] -= q[IBY] * q[IBX];
    flux[IV] -= q[IBY] * q[IBY] - pmag;
    flux[IW] -= q[IBY] * q[IBZ];
    flux[IE] -= q[IBY] * udotB - q[IV] * pmag; // Pareil ici, pmag ? / Update : non-nécessaire car énergie mag déja contenue dans u[IE]

    // GLM Flux
    // flux[IBY] += ch*ch * q[IPSI];
    flux[IBY] = q[IPSI];
    flux[IPSI] = ch*ch
    return primToCons(q, params); // Important point as it recomputes the total energy !
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

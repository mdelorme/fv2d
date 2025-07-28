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
  
    if (dir == IX){
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
   * @brief Reflecting boundary condition in hydrostatic equilibrium
   **/
   KOKKOS_INLINE_FUNCTION
   State fillHSE(Array Q, int i, int j, IDir dir, const DeviceParams &params, real_t t) {
    if (dir == IX) {
      Kokkos::abort("ERROR : Cannot use hse boundary conditions along X");
    }
    else {
      // ymin
      if (j < params.jbeg) {
        State out = getStateFromArray(Q, i, params.jbeg);
        real_t rho = out[IR];
        real_t dy  = params.dy * (params.jbeg - j);
        real_t p   = out[IP] - dy * rho * getGravity(i, j, IY, params);
        out[IP] = p;

        if (params.perturbation) {
          out[IV] = params.perturb_A * Kokkos::sin(12 * M_PI * t / params.perturb_tf);
        }
        else {
          out[IV] *= -1.0;
        }

        return out;
      }
      // ymax
      else {
        State out = getStateFromArray(Q, i, params.jend-1);
        real_t rho = out[IR];
        real_t dy  = params.dy * (j - params.jend+1);
        real_t p   = out[IP] + dy * rho * getGravity(i, j, IY, params);        
        out[IP] = p;
        
        out[IV] *= -1.0;
        
        return out;
      }
    }
   }
} // anonymous namespace


class BoundaryManager {
public:
  Params full_params;

  BoundaryManager(const Params &full_params) 
    : full_params(full_params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q, real_t t) {
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
                                default:
                                case BC_ABSORBING:      return fillAbsorbing(Q, iref, j); break;
                                case BC_REFLECTING:     return fillReflecting(Q, i, j, iref, j, IX, params); break;
                                case BC_PERIODIC:       return fillPeriodic(Q, i, j, IX, params); break;
                                case BC_HSE:            return fillHSE(Q, i, j, IX, params, t); break;
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
                                default:
                                case BC_ABSORBING:      return fillAbsorbing(Q, i, jref); break;
                                case BC_REFLECTING:     return fillReflecting(Q, i, j, i, jref, IY, params); break;
                                case BC_PERIODIC:       return fillPeriodic(Q, i, j, IY, params); break;
                                case BC_HSE:            return fillHSE(Q, i, j, IY, params, t); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}
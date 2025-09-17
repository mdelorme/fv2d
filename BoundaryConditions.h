#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"

namespace fv2d {
  namespace {

  KOKKOS_INLINE_FUNCTION
  void applyMagneticBoundaries(State &q, IDir dir, const DeviceParams &params){
    if (dir == IX){
      switch (params.magnetic_boundary_x){
          case BCMAG_NORMAL_FIELD:{
            q[IBY] = 0.0;
            q[IBZ] = 0.0;
            break;
          }
          case BCMAG_PERFECT_CONDUCTOR: q[IBX] = 0.0; break;
          default: break; // SAME_AS_HYDRO
        }
      }
    if (dir == IY){
    switch (params.magnetic_boundary_y){
        case BCMAG_NORMAL_FIELD:{
          q[IBX] = 0;
          q[IBZ] = 0;
          break;
        }
        case BCMAG_PERFECT_CONDUCTOR: q[IBY] = 0.0; break;
        default: break; // SAME_AS_HYDRO
      }
    }
  }
  /**
   * @brief Absorbing conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillAbsorbing(Array Q, int iref, int jref, IDir dir, const DeviceParams &params) {
    State q = getStateFromArray(Q, iref, jref);
    #ifdef MHD
      applyMagneticBoundaries(q, dir, params);
    #endif // MHD
    return q;
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
    #ifdef MHD
      applyMagneticBoundaries(q, dir, params);
    #endif //MHD
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
    State q = getStateFromArray(Q, i, j);
    #ifdef MHD
      applyMagneticBoundaries(q, dir, params);
    #endif
    return q;
  }

  /* Magnetic Boundaries: 
  Can be of type : 
    - SAME_AS_HYDRO: in this case we can just call the pre-existing hydro boundary conditions with a MHD State and do nothing special
    - NORMAL_FIELD: in this case only the components of B are null except for the normal one (with respect to the boundary considered)
    - PERFECT_CONDUCTOR: in this case, the component normal to the boundary is 0 and others remain untouched
  */
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
                                default:
                                case BC_ABSORBING:  return fillAbsorbing(Q, iref, j, IX, params); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, iref, j, IX, params); break;
                                case BC_PERIODIC:   return fillPeriodic(Q, i, j, IX, params); break;
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
                                case BC_ABSORBING:  return fillAbsorbing(Q, i, jref, IY, params); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, i, jref, IY, params); break;
                                case BC_PERIODIC:   return fillPeriodic(Q, i, j, IY, params); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}
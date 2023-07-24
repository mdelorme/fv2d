#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"
#include "Geometry.h"

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
  State fillReflecting(Array Q, int i, int j, int iref, int jref, IDir dir, const Params &params) {
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
  
    if (dir == IX)
      q[IU] *= -1.0;
    else
      q[IV] *= -1.0;

    return q;
  }

  /**
   * @brief Geometry 'ring_y' reflecting
   */
  KOKKOS_INLINE_FUNCTION
  State fillReflectingRingY(Array Q, int i, int j, int iref, int jref, IDir dir, const Params &params, const Geometry &geo) {
    int isym, jsym;
    assert(dir == IY);

    int jpiv = (j < jref ? params.jbeg : params.jend);
    isym = i;
    jsym = 2*jpiv - j - 1;

    State q = getStateFromArray(Q, isym, jsym);

    Pos p = geo.mapc2p_center(isym, jsym);
    real_t cos2x, sin2x;
    {
      real_t norm = sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
      real_t cos = p[IX] / norm;
      real_t sin = p[IY] / norm;
      cos2x = cos*cos - sin*sin;
      sin2x = 2*sin*cos;
    }
  
    q[IU] = -cos2x * q[IU] - sin2x * q[IV];
    q[IV] = -sin2x * q[IU] + cos2x * q[IV];

    return q;
  }

  /**
   * @brief Periodic boundary conditions
   * 
   */
  KOKKOS_INLINE_FUNCTION
  State fillPeriodic(Array Q, int i, int j, IDir dir, const Params &params) {
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
} // anonymous namespace


class BoundaryManager {
public:
  Params params;
  Geometry geometry;

  BoundaryManager(const Params &params) 
    : params(params),
      geometry(params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q) {
    auto bc_x = params.boundary_x;
    auto bc_y = params.boundary_y;
    auto params = this->params;
    auto geometry = this->geometry;

    Kokkos::parallel_for( "Filling X-boundary",
                          params.range_xbound,
                          KOKKOS_LAMBDA(int i, int j) {

                            int ileft     = i;
                            int iright    = params.iend+i;
                            int iref_left = params.ibeg;
                            int iref_right = params.iend-1;

                            auto fill = [&](int i, int iref) {
                              switch (bc_x) {
                                default:
                                case BC_ABSORBING:  return fillAbsorbing(Q, iref, j); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, iref, j, IX, params); break;
                                case BC_PERIODIC:   return fillPeriodic(Q, i, j, IX, params); break;
                              }
                            };

                            setStateInArray(Q, ileft, j,  fill(ileft, iref_left));
                            setStateInArray(Q, iright, j, fill(iright, iref_right));
                          });

    Kokkos::parallel_for( "Filling Y-boundary",
                          params.range_ybound,
                          KOKKOS_LAMBDA(int i, int j) {

                            int jtop     = j;
                            int jbot     = params.jend+j;
                            int jref_top = params.jbeg;
                            int jref_bot = params.jend-1;

                            auto fill = [&](int j, int jref) {
                              switch (bc_y) {
                                default:
                                case BC_ABSORBING:  return fillAbsorbing(Q, i, jref); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, i, jref, IY, params); break;
                                case BC_PERIODIC:   return fillPeriodic(Q, i, j, IY, params); break;
                                case BC_REFLECTING_RING_Y:   return fillReflectingRingY(Q, i, j, i, jref, IY, params, geometry); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}
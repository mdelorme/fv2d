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
  
    if (dir == IX)
      q[IU] *= -1.0;
    else
      q[IV] *= -1.0;

    return q;
  }

  /**
   * @brief Geometry radial reflecting
   */
  KOKKOS_INLINE_FUNCTION
  State fillRadialReflecting(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params, const Geometry &geo) {
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
    Pos p = geo.mapc2p_center(isym, jsym);

    const real_t normal_vel = q[IU]*p[IX] + q[IV]*p[IY];
    q[IU] = q[IU] - 2*normal_vel*p[IX];
    q[IV] = q[IV] - 2*normal_vel*p[IY];

    // real_t cos2x, sin2x;
    // {
    //   real_t norm = sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
    //   real_t cos = p[IX] / norm;
    //   real_t sin = p[IY] / norm;
    //   cos2x = cos*cos - sin*sin;
    //   sin2x = 2*sin*cos;
    // }
  
    // q[IU] = -cos2x * q[IU] - sin2x * q[IV];
    // q[IV] = -sin2x * q[IU] + cos2x * q[IV];

    return q;
  }

  /**
   * @brief Fixed boudary for radial grid by reading value from spline file, and set radial velocity to 0
   */
  KOKKOS_INLINE_FUNCTION
  State fillFixedReadfile(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params, const Geometry &geo) {
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

    // State q;
    State q = getStateFromArray(Q, isym, jsym);
    Pos p = geo.mapc2p_center(i, j);

    const real_t r = norm(p);
    // const real_t r = params.radial_radius;
    
    p = p / norm(p);
    const real_t normal_vel = q[IU]*p[IX] + q[IV]*p[IY];
    

    q[IR] = params.spl_rho(r);
    q[IU] = q[IU] - 2*normal_vel*p[IX];
    q[IV] = q[IV] - 2*normal_vel*p[IY];
    // q[IU] = 0;
    // q[IV] = 0;
    q[IP] = params.spl_prs(r);

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

  KOKKOS_INLINE_FUNCTION
  State fillIsothermalDirichlet(Array Q, int i, int j, IDir dir, const DeviceParams &params, const Geometry &geo) {
    State q{0};

    const real_t g = Kokkos::max(params.gx, params.gy);

    auto [x, y] = geo.mapc2p_center(i,j);
    real_t rho0 = 1.2;
    real_t p0 = 1.0;
    real_t phi = 0.5 * (x*x + y*y) * g;

    q[IR] = rho0 * exp(-rho0 * phi / p0);
    q[IP] = p0 * exp(-rho0 * phi / p0);

    return q;
  }
} // anonymous namespace


class BoundaryManager {
public:
  Params full_params;
  Geometry geometry;

  BoundaryManager(const Params &full_params) 
    : full_params(full_params),
      geometry(full_params.device_params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q) {
    auto params = full_params.device_params;
    auto bc_x = params.boundary_x;
    auto bc_y = params.boundary_y;
    auto geometry = this->geometry;

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
                                case BC_ABSORBING:         return fillAbsorbing(Q, iref, j); break;
                                case BC_REFLECTING:        return fillReflecting(Q, i, j, iref, j, IX, params); break;
                                case BC_PERIODIC:          return fillPeriodic(Q, i, j, IX, params); break;
                                case BC_RADIAL_REFLECTING: return fillRadialReflecting(Q, i, j, iref, j, IX, params, geometry); break;
                                case BC_FIXED_READFILE:    return fillFixedReadfile(Q, i, j, iref, j, IX, params, geometry); break;
                                // case BC_FIXED_READFILE:    return fillFixedReadfile(Q, params); break;
                                case BC_ISOTHERMAL_DIRICHLET: return fillIsothermalDirichlet(Q, i, j, IX, params, geometry); break;
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
                                case BC_ABSORBING:         return fillAbsorbing(Q, i, jref); break;
                                case BC_REFLECTING:        return fillReflecting(Q, i, j, i, jref, IY, params); break;
                                case BC_PERIODIC:          return fillPeriodic(Q, i, j, IY, params); break;
                                case BC_RADIAL_REFLECTING: return fillRadialReflecting(Q, i, j, i, jref, IY, params, geometry); break;
                                case BC_FIXED_READFILE:    return fillFixedReadfile(Q, i, j, i, jref, IY, params, geometry); break;
                                // case BC_FIXED_READFILE:    return fillFixedReadfile(Q, params); break;
                                case BC_ISOTHERMAL_DIRICHLET: return fillIsothermalDirichlet(Q, i, j, IY, params, geometry); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
                          });
  }
};

}
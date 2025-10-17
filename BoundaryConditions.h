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

  KOKKOS_INLINE_FUNCTION
  State getBoundaryFlux(const State& q_in, State &flux_in, int i, int j, IDir dir, const DeviceParams &params){
    State q_out = q_in;
    const bool is_left_boundary   = i == params.ibeg;
    const bool is_right_boundary  = i == params.iend;
    const bool is_bottom_boundary = j == params.jbeg;
    const bool is_upper_boundary  = j == params.jend-1;
    // TODO: no C array -> change to kokkos array
    const bool is_boundary[2] = {is_left_boundary || is_right_boundary, is_bottom_boundary || is_upper_boundary};
    const BoundaryType bc_type[2] = {params.boundary_x, params.boundary_y};
    const MagneticBoundaryType bc_mag_type[2] = {params.magnetic_boundary_x, params.magnetic_boundary_y};
    // TODO: Réadapter ces conditions pour fv2d
    const bool reflecting = (is_boundary[dir] && bc_type[dir] == BC_REFLECTING);
    const bool absorbing  = (is_boundary[dir] && bc_type[dir] == BC_ABSORBING);
    const bool periodic   = (is_boundary[dir] && bc_type[dir] == BC_PERIODIC);

    const bool mag_perfect_conductor = (is_boundary[dir] && bc_mag_type[dir] == BCMAG_PERFECT_CONDUCTOR);
    const bool mag_normal_field = (is_boundary[dir] && bc_mag_type[dir] == BCMAG_NORMAL_FIELD);

    // Vect B_out = B_in;
    if( reflecting )
      {
        q_out[IU] = (dir == IX ? 0.0 : q_in[IU]);
        q_out[IV] = (dir == IY ? 0.0 : q_in[IV]);
        q_out[IW] = (dir == IZ ? 0.0 : q_in[IW]);

        q_out[IBX] = (dir == IX ? 0.0 : q_in[IBX]);
        q_out[IBY] = (dir == IY ? 0.0 : q_in[IBY]);
        q_out[IBZ] = (dir == IZ ? 0.0 : q_in[IBZ]);
      }
    else if( absorbing )
      {
        // Do nothing, q_out = q_in
      }
    
    if (mag_perfect_conductor){
      q_out[IBX] = (dir == IX ? 0.0 : q_in[IBX]);
      q_out[IBY] = (dir == IY ? 0.0 : q_in[IBY]);
      q_out[IBZ] = (dir == IZ ? 0.0 : q_in[IBZ]);
    }

    else if (mag_normal_field){
      q_out[IBX] = (dir == IX ? q_in[IBX] : 0.0);
      q_out[IBY] = (dir == IY ? q_in[IBY] : 0.0);
      q_out[IBZ] = (dir == IZ ? q_in[IBZ] : 0.0);
    }

    const State u_out = primToCons(q_out, params);

    const real_t vx = q_out[IU], vy = q_out[IV], vz = q_out[IW];
    const Vect v_out {vx, vy, vz}; 
    const real_t v_normal = v_out[dir];

    const real_t Bx = q_out[IBX], By = q_out[IBY], Bz = q_out[IBZ];
    const Vect B_out {Bx, By, Bz};
    const real_t B_normal = B_out[dir];

    const real_t p_gas = q_out[IP];
    const real_t p_mag = 0.5 * (Bx*Bx + By*By + Bz*Bz);

    State flux_out;
    flux_out[IR] = u_out[IR] * v_normal;
    flux_out[IU] = u_out[IU] * v_normal - Bx * B_normal + (dir == IX ? p_gas + p_mag : 0.0);
    flux_out[IV] = u_out[IV] * v_normal - By * B_normal + (dir == IY ? p_gas + p_mag : 0.0);
    flux_out[IW] = u_out[IW] * v_normal - Bz * B_normal + (dir == IZ ? p_gas + p_mag : 0.0);

    flux_out[IE] = (u_out[IE] + p_gas + p_mag) * v_normal - B_normal * dot(B_out, v_out);

    flux_out[IBX] = Bx * v_normal - B_normal * vx;
    flux_out[IBY] = By * v_normal - B_normal * vy;
    flux_out[IBZ] = Bz * v_normal - B_normal * vz;

    return flux_out;
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
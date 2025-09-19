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
  State getBoundaryFlux(State q_in, State &flux_in, int i, int j, IDir dir, const DeviceParams &params){
    const State u_in = consToPrim(q_in, params);
    const bool is_left_boundary   = i == params.ibeg;
    const bool is_right_boundary  = i == params.iend;
    const bool is_bottom_boundary = j == params.jbeg;
    const bool is_upper_boundary  = j == params.jend-1;
    // TODO: no C array -> change to kokkos array
    // const bool min_boundary[2] = {is_left_boundary, is_bottom_boundary};
    // const bool max_boundary[2] = {is_right_boundary, is_upper_boundary};
    const bool is_boundary[2] = {is_left_boundary || is_right_boundary, is_bottom_boundary || is_upper_boundary};
    const BoundaryType bc_type[2] = {params.boundary_x, params.boundary_y};
    const MagneticBoundaryType bc_mag_type[2] = {params.magnetic_boundary_x, params.magnetic_boundary_y};
    // TODO: Réadapter ces conditions pour fv2d
    const bool reflecting = (is_boundary[dir] && bc_type[dir] == BC_REFLECTING);
    const bool absorbing  = (is_boundary[dir] && bc_type[dir] == BC_ABSORBING);
    const bool periodic   = (is_boundary[dir] && bc_type[dir] == BC_PERIODIC);

    const bool mag_perfect_conductor = (is_boundary[dir] && bc_mag_type[dir] == BCMAG_PERFECT_CONDUCTOR);
    const bool mag_normal_field = (is_boundary[dir] && bc_mag_type[dir] == BCMAG_NORMAL_FIELD);


    Vect v_in {q_in[IU], q_in[IV], q_in[IW]};
    Vect B_in {q_in[IBX], q_in[IBY], q_in[IBZ]};
    // Vect B_out = B_in;
    
    Vect p_in {0.0, 0.0, 0.0};
    p_in[dir] = q_in[IP];
    
    State flux_out = flux_in;
    // /**
    //  * In the reflecting case, the values in the "ghosts" are supposed to be reflecting the
    //  * ones inside the domain, hence reconstruction yields u_norm = 0 at the boundary, simplifying
    //  * the calculation of the flux to only the pressure gradient term in the flux.
    //  */
    if( reflecting )
    {
      v_in[dir] *= -1.0;
      flux_out[IU] = p_in[IX];
      flux_out[IV] = p_in[IY];
      flux_out[IW] = p_in[IZ];
      B_in[dir] *= -1.0;
    }
    /**
     * In the absorbing case, the values in the ghosts are supposed to be interpolated from the ones 
     * inside the domain to provide a null gradient through the boundary. Hence we can take the
     * reconstructed value at the boundary as the riemann-problem solution.
     */
    else if( absorbing )
    {
      real_t f_rho = q_in[IR]*v_in[dir];
      
      flux_out[IR] = f_rho;
      flux_out[IU] = f_rho*q_in[IU] + p_in[IX];
      flux_out[IV] = f_rho*q_in[IV] + p_in[IY];
      flux_out[IW] = f_rho*q_in[IW] + p_in[IZ];
      flux_out[IE] = (q_in[IP] + u_in[IE]) * v_in[dir];
    }
    else
    {
      flux_out = flux_in;
    }
    
    /**
     * Magnetic boundaries
     * 
     * Perfect conductor : Bz = 0
     */
    Vect B_out {};
    if ( mag_perfect_conductor )
    { 
      B_out[dir] = 0.0;
    }
    // /**
    //  * Normal field : Bx=By=0
    //  */
    else if ( mag_normal_field )
    {
      B_out[dir] = B_in[dir];
    }
    else
    {
      B_out = B_in;
    }

    Vect pmag {0.0, 0.0, 0.0};
    pmag[dir] = 0.5 * dot(B_out, B_out); // 1/2 B^2
    flux_out[IU] -= B_out[IX] * B_out[dir] - pmag[IX];
    flux_out[IV] -= B_out[IY] * B_out[dir] - pmag[IY];
    flux_out[IW] -= B_out[IZ] * B_out[dir] - pmag[IZ];
    flux_out[IE] -= B_out[dir] * dot(v_in, B_out) - v_in[dir] * pmag[dir];

    flux_out[IBX] = B_out[IX] * v_in[dir] - B_out[dir] * q_in[IU];
    flux_out[IBY] = B_out[IY] * v_in[dir] - B_out[dir] * q_in[IV];
    flux_out[IBZ] = B_out[IZ] * v_in[dir] - B_out[dir] * q_in[IW];

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
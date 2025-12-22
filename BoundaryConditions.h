#pragma once

#include <cassert>
#include <map>

#include "SimInfo.h"

namespace fv2d
{
namespace
{

#ifdef MHD
KOKKOS_INLINE_FUNCTION
void applyMagneticBoundaries(State &q, IDir dir, const DeviceParams &params)
{
  if (dir == IX)
  {
    switch (params.magnetic_boundary_x)
    {
    case BCMAG_NORMAL_FIELD:
    {
      q[IBY] *= -1.0;
      q[IBZ] *= -1.0;
      break;
    }
    case BCMAG_PERFECT_CONDUCTOR:
      q[IBX] *= -1.0;
      break;
    default:
      break; // SAME_AS_HYDRO
    }
  }
  if (dir == IY)
  {
    switch (params.magnetic_boundary_y)
    {
    case BCMAG_NORMAL_FIELD:
    {
      q[IBX] *= -1.0;
      q[IBZ] *= -1.0;
      break;
    }
    case BCMAG_PERFECT_CONDUCTOR:
      q[IBY] *= -1.0;
      break;
    default:
      break; // SAME_AS_HYDRO
    }
  }
}

KOKKOS_INLINE_FUNCTION
State getBoundaryFlux(const State &q_in, int i, int j, IDir dir, const real_t c_h, const DeviceParams &params)
{
  State q_out                               = q_in;
  const BoundaryType bc_type[2]             = {params.boundary_x, params.boundary_y};
  const MagneticBoundaryType bc_mag_type[2] = {params.magnetic_boundary_x, params.magnetic_boundary_y};
  // TODO: Réadapter ces conditions pour fv2d
  const bool reflecting = (bc_type[dir] == BC_REFLECTING);
  const bool absorbing  = (bc_type[dir] == BC_ABSORBING);

  const bool mag_perfect_conductor = (bc_mag_type[dir] == BCMAG_PERFECT_CONDUCTOR);
  const bool mag_normal_field      = (bc_mag_type[dir] == BCMAG_NORMAL_FIELD);

  // Vect B_out = B_in;
  if (reflecting)
  {
    q_out[IU] = (dir == IX ? 0.0 : q_in[IU]);
    q_out[IV] = (dir == IY ? 0.0 : q_in[IV]);
    q_out[IW] = (dir == IZ ? 0.0 : q_in[IW]);

    q_out[IBX] = (dir == IX ? 0.0 : q_in[IBX]);
    q_out[IBY] = (dir == IY ? 0.0 : q_in[IBY]);
    q_out[IBZ] = (dir == IZ ? 0.0 : q_in[IBZ]);
  }
  else if (absorbing)
  {
    // Do nothing, q_out = q_in
  }

  if (mag_perfect_conductor)
  {
    q_out[IBX] = (dir == IX ? 0.0 : q_in[IBX]);
    q_out[IBY] = (dir == IY ? 0.0 : q_in[IBY]);
    q_out[IBZ] = (dir == IZ ? 0.0 : q_in[IBZ]);
  }

  else if (mag_normal_field)
  {
    q_out[IBX] = (dir == IX ? q_in[IBX] : 0.0);
    q_out[IBY] = (dir == IY ? q_in[IBY] : 0.0);
    q_out[IBZ] = (dir == IZ ? q_in[IBZ] : 0.0);
  }

  const State u_out = primToCons(q_out, params);

  const real_t vx = q_out[IU], vy = q_out[IV], vz = q_out[IW];
  const Vect v_out{vx, vy, vz};
  const real_t v_normal = v_out[dir];

  const real_t Bx = q_out[IBX], By = q_out[IBY], Bz = q_out[IBZ];
  const Vect B_out{Bx, By, Bz};
  const real_t B_normal = B_out[dir];

  const real_t p_gas = q_out[IP];
  const real_t p_mag = 0.5 * (Bx * Bx + By * By + Bz * Bz);

  State flux_out;
  flux_out[IR] = u_out[IR] * v_normal;
  flux_out[IU] = u_out[IU] * v_normal - Bx * B_normal + (dir == IX ? p_gas + p_mag : 0.0);
  flux_out[IV] = u_out[IV] * v_normal - By * B_normal + (dir == IY ? p_gas + p_mag : 0.0);
  flux_out[IW] = u_out[IW] * v_normal - Bz * B_normal + (dir == IZ ? p_gas + p_mag : 0.0);

  flux_out[IE] = (u_out[IE] + p_gas + p_mag) * v_normal - B_normal * dot(B_out, v_out);

  flux_out[IBX] = Bx * v_normal - B_normal * vx;
  flux_out[IBY] = By * v_normal - B_normal * vy;
  flux_out[IBZ] = Bz * v_normal - B_normal * vz;

  if (params.riemann_solver == IDEALGLM || params.div_cleaning == DEDNER)
  {
    real_t Bxm     = B_normal;
    real_t psi_m   = q_in[IPSI];
    flux_out[IBX]  = (dir == IX ? psi_m : 0.0);
    flux_out[IBY]  = (dir == IY ? psi_m : 0.0);
    flux_out[IBZ]  = (dir == IZ ? psi_m : 0.0);
    flux_out[IPSI] = c_h * c_h * Bxm;
  }
  return flux_out;
}
#endif // MHD

/**
 * @brief Absorbing conditions
 */
KOKKOS_INLINE_FUNCTION
State fillAbsorbing(Array Q, int iref, int jref, IDir dir, const DeviceParams &params)
{
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
State fillReflecting(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params)
{
  int isym, jsym;
  if (dir == IX)
  {
    int ipiv = (i < iref ? params.ibeg : params.iend);
    isym     = 2 * ipiv - i - 1;
    jsym     = j;
  }
  else
  {
    int jpiv = (j < jref ? params.jbeg : params.jend);
    isym     = i;
    jsym     = 2 * jpiv - j - 1;
  }

  State q = getStateFromArray(Q, isym, jsym);

  if (dir == IX)
  {
    q[IU] *= -1.0;
#ifdef MHD
    q[IBX] *= -1.0;
#endif
  }
  else
  {
    q[IV] *= -1.0;
#ifdef MHD
    q[IBY] *= -1.0;
#endif
  }
#ifdef MHD
  applyMagneticBoundaries(q, dir, params);
#endif // MHD
  return q;
}

/**
 * @brief Periodic boundary conditions
 *
 */
KOKKOS_INLINE_FUNCTION
State fillPeriodic(Array Q, int i, int j, IDir dir, const DeviceParams &params)
{
  if (dir == IX)
  {
    if (i < params.ibeg)
      i += params.Nx;
    else
      i -= params.Nx;
  }
  else
  {
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
  - SAME_AS_HYDRO: in this case we can just call the pre-existing hydro boundary conditions with a MHD State and do
nothing special
  - NORMAL_FIELD: in this case only the components of B are null except for the normal one (with respect to the boundary
considered)
  - PERFECT_CONDUCTOR: in this case, the component normal to the boundary is 0 and others remain untouched
*/
/*
 * Vertical boundary condtions for the Cattaneo (1991) setup
 */
KOKKOS_INLINE_FUNCTION
State fillC91(Array Q, int i, int j, int iref, int jref, IDir dir, const DeviceParams &params)
{
  const real_t dy = params.dy;
  const int jpiv  = (j < jref ? params.jbeg : params.jend);
  const int jsym  = 2 * jpiv - j - 1;

  const bool is_upper_bc = (j < jref);
  // const bool is_upper_bc = (j < params.jend);
  // const int offset = (is_lower_bc ? j - params.jbeg : j - params.jend);
  Pos pos  = getPos(params, i, j);
  real_t y = pos[IY];

  // real_t T   = (1.0 + params.theta1 * y);
  // real_t rho = pow(T, params.m1);
  // real_t prs = pow(T, params.m1 + 1.0);

  State q_dc = getStateFromArray(Q, i, jsym);
  State q_gc = q_dc;

  // q_gc[IR] = rho;
  // q_gc[IP] = prs;
  // q_gc[IU] = 0.0;
  // q_gc[IV] = 0.0;
#ifdef MHD
  q_gc[IBX]         = -q_dc[IBX];
  q_gc[IBY]         = q_dc[IBY];
  q_gc[IBZ]         = -q_dc[IBZ];
  const real_t pmag = 0.5 * (q_gc[IBX] * q_gc[IBX] + q_gc[IBY] * q_gc[IBY] + q_gc[IBZ] * q_gc[IBZ]);
#else
  const real_t pmag = 0.0;
#endif // MHD

  if (is_upper_bc)
  {
    // Note: 2025-12-21 : Attention, je calcule que les valeurs de la première cellule GC pour l'instant
    // We designate with _p the cells in the domain, and _m outisde the domain.
    // The number refers to the cell's lace with respect to the inteface.
    real_t rho_p1, rho_p2, p_p1, p_p2, p_m1, rho_m1;

    rho_p1 = Q(params.Ng, i, IR);
    rho_p2 = Q(params.Ng + 1, i, IR);
    rho_m1 = 2.0 * rho_p1 - rho_p2;
    p_p1   = Q(params.Ng, i, IP);

    if (j == 1)
    {
    } // Values are first computed for the first GC

    if (j == 0) // Extrapolate values from the first GC
    {
      rho_p2 = rho_p1;
      rho_p1 = rho_m1;
      rho_m1 = 2.0 * rho_p1 - rho_p2;
      p_p1   = Q(1, i, IP);
    }

    p_m1     = p_p1 - params.gy * dy * (5.0 * rho_m1 + 8.0 * rho_p1 - rho_p2) / 12.0; // Zingale (2002), eq. (50).
    q_gc[IV] = -q_dc[IV];
    q_gc[IR] = rho_m1;
    q_gc[IP] = p_m1 + pmag;
  }
  else
  {
    // printf("params.Nty=%d, j=%d, params.jend=%d\n", params.Nty, j, params.jend);
    // params.Nty=132, j=131, params.jend=130 ->  Donc jend = j_GC1, jend + 1 = j_GC2
    real_t rho_m1, rho_m2, p_m1, p_p1, rho_p1;
    const int j_dc = params.jend - 1; // Last domin cell
    rho_m1         = Q(j_dc, i, IR);
    rho_m2         = Q(j_dc - 1, i, IR);
    rho_p1         = 2.0 * rho_m1 - rho_m2;
    p_m1           = Q(j_dc, i, IP);

    if (j == params.jend)
    {
    } // First ghost cell, nothing more to do

    if (j == params.jend + 1)
    {
      rho_m2 = rho_m1;
      rho_m1 = rho_p1;
      rho_p1 = 2.0 * rho_m1 - rho_m2;
      p_m1   = Q(j_dc + 1, i, IP);
    }
    p_p1 = p_m1 + params.gy * dy * (5.0 * rho_p1 + 8.0 * rho_m1 - rho_m2) / 12.0; // Zingale (2002), eq. (41).

    q_gc[IV] = -q_dc[IV];
    q_gc[IR] = rho_p1;
    q_gc[IP] = p_p1 + pmag;
  }
  return q_gc;
}
} // anonymous namespace

class BoundaryManager
{
public:
  Params full_params;

  BoundaryManager(const Params &full_params) : full_params(full_params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q)
  {
    auto params = full_params.device_params;
    auto bc_x   = params.boundary_x;
    auto bc_y   = params.boundary_y;

    Kokkos::parallel_for(
        "Filling X-boundary",
        full_params.range_xbound,
        KOKKOS_LAMBDA(int i, int j) {
          int ileft      = i;
          int iright     = params.iend + i;
          int iref_left  = params.ibeg;
          int iref_right = params.iend - 1;

          auto fill = [&](int i, int iref)
          {
            switch (bc_x)
            {
            default:
            case BC_ABSORBING:
              return fillAbsorbing(Q, iref, j, IX, params);
              break;
            case BC_REFLECTING:
              return fillReflecting(Q, i, j, iref, j, IX, params);
              break;
            case BC_PERIODIC:
              return fillPeriodic(Q, i, j, IX, params);
              break;
            }
          };

          setStateInArray(Q, ileft, j, fill(ileft, iref_left));
          setStateInArray(Q, iright, j, fill(iright, iref_right));
        });

    Kokkos::parallel_for(
        "Filling Y-boundary",
        full_params.range_ybound,
        KOKKOS_LAMBDA(int i, int j) {
          int jtop     = j;
          int jbot     = params.jend + j;
          int jref_top = params.jbeg;
          int jref_bot = params.jend - 1;

          auto fill = [&](int j, int jref)
          {
            switch (bc_y)
            {
            default:
            case BC_ABSORBING:
              return fillAbsorbing(Q, i, jref, IY, params);
              break;
            case BC_REFLECTING:
              return fillReflecting(Q, i, j, i, jref, IY, params);
              break;
            case BC_PERIODIC:
              return fillPeriodic(Q, i, j, IY, params);
              break;
            case BC_C91:
              return fillC91(Q, i, j, i, jref, IY, params);
            }
          };

          setStateInArray(Q, i, jtop, fill(jtop, jref_top));
          setStateInArray(Q, i, jbot, fill(jbot, jref_bot));
        });
  }
};

} // namespace fv2d

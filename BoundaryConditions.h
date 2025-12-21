#pragma once

#include <cassert>
#include <map>

#include "SimInfo.h"

namespace fv2d
{
namespace
{
/**
 * @brief Absorbing conditions
 */
KOKKOS_INLINE_FUNCTION
State fillAbsorbing(Array Q, int iref, int jref) { return getStateFromArray(Q, iref, jref); };

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
    q[IU] *= -1.0;
  else
    q[IV] *= -1.0;

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

  return getStateFromArray(Q, i, j);
}

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
    p_m1 = p_p1 - params.gy * dy * (5.0 * rho_m1 + 8.0 * rho_p1 - rho_p2) / 12.0; // Zingale (2002), eq. (50).

    q_gc[IV] = -q_dc[IV];
    q_gc[IR] = rho_m1;
    q_gc[IP] = p_m1;
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
    q_gc[IP] = p_p1;
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
              return fillAbsorbing(Q, iref, j);
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
              return fillAbsorbing(Q, i, jref);
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

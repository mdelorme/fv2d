#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"

namespace fv1d {

State fill_absorbing(Array &Q, int i, int j, int iref, int jref, real_t dt, IDir dir) {
  return Q[jref][iref];
}

State fill_reflecting(Array &Q, int i, int j, int iref, int jref, real_t dt, IDir dir) {
  int isym, jsym;
  if (dir == IX) {
    int ipiv = (i < iref ? ibeg : iend);
    isym = 2*ipiv - i - 1;
    jsym = j;  
  }
  else {
    int jpiv = (j < jref ? jbeg : jend);
    isym = i;
    jsym = 2*jpiv - j - 1;
  }

  State q = Q[jsym][isym];
  
  if (dir == IX)
    q[IU] *= -1.0;
  else
    q[IV] *= -1.0;

  return q;
}

State fill_periodic(Array &Q, int i, int j, int iref, int jref, real_t dt, IDir dir) {
  if (dir == IX) {
    if (i < ibeg)
      i += Nx;
    else
      i -= Nx;
  }
  else {
    if (j < jbeg)
      j += Ny;
    else
      j -= Ny;
  }

  return Q[j][i];
}

/**
 * One time initialization for the boundary conditions
 **/
void init_boundaries() {
  assert(Ng == 2); // Most functions have been coded with Ng = 2 !!!!

  switch (boundary_x) {
    case BC_ABSORBING: bc_function_x = fill_absorbing; break;
    case BC_PERIODIC:  bc_function_x = fill_periodic;  break;
    default: bc_function_x = fill_reflecting; break;
  }

  switch (boundary_y) {
    case BC_ABSORBING: bc_function_y = fill_absorbing; break;
    case BC_PERIODIC:  bc_function_y = fill_periodic;  break;
    default: bc_function_y = fill_reflecting; break;
  }}

/**
 * Fills the boundary with a value. The value is returned from a user-defed method.
 * Note that this function is overly complicated for this type of code.
 * There is a real reason to that: to replicate the way it is called in dyablo
 */
void fill_boundaries(Array &Q, real_t dt) {
  // Filling boundaries along X
  for (int j=jbeg; j < jend; ++j) {
    for (int i=0; i < Ng; ++i) {
      int ileft      = i;
      int iright     = iend+i;
      int iref_left  = ibeg;
      int iref_right = iend-1;

      // And apply
      Q[j][ileft]  = bc_function_x(Q, ileft,  j, iref_left,  j, dt, IX);
      Q[j][iright] = bc_function_x(Q, iright, j, iref_right, j, dt, IX);
    }
  }

  for (int j=0; j < Ng; ++j) {
    for (int i=0; i < Ntx; ++i) {
      int jtop = j;
      int jbot = jend+j;
      int jref_top = jbeg;
      int jref_bot = jend-1;

      // And apply
      Q[jtop][i] = bc_function_y(Q, i, jtop, i, jref_top, dt, IY);
      Q[jbot][i] = bc_function_y(Q, i, jbot, i, jref_bot, dt, IY);
    }
  }
}


}
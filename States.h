#pragma once

namespace fv2d {

KOKKOS_INLINE_FUNCTION
State getStateFromArray(Array arr, int i, int j) {
  return {arr(j, i, IR),
          arr(j, i, IU),
          arr(j, i, IV),
          arr(j, i, IP)};
} 

KOKKOS_INLINE_FUNCTION
void setStateInArray(Array arr, int i, int j, State st) {
  for (int ivar=0; ivar < Nfields; ++ivar)
    arr(j, i, ivar) = st[ivar];
}

KOKKOS_INLINE_FUNCTION
State primToCons(State &q, const Params &params) {
  State res;
  res[IR] = q[IR];
  res[IU] = q[IR]*q[IU];
  res[IV] = q[IR]*q[IV];

  real_t Ek = 0.5 * (res[IU]*res[IU] + res[IV]*res[IV]) / q[IR];
  res[IE] = (Ek + q[IP] / (params.gamma0-1.0));
  return res;
}


KOKKOS_INLINE_FUNCTION
State consToPrim(State &u, const Params &params) {
  State res;
  res[IR] = u[IR];
  res[IU] = u[IU] / u[IR];
  res[IV] = u[IV] / u[IR];

  real_t Ek = 0.5 * res[IR] * (res[IU]*res[IU] + res[IV]*res[IV]);
  res[IP] = (u[IE] - Ek) * (params.gamma0-1.0);
  return res; 
}

KOKKOS_INLINE_FUNCTION
real_t speedOfSound(State &q, const Params &params) {
  return sqrt(q[IP] * params.gamma0 / q[IR]);
}

KOKKOS_INLINE_FUNCTION
State& operator+=(State &a, State b) {
  for (int i=0; i < Nfields; ++i)
    a[i] += b[i];
  return a;
}

KOKKOS_INLINE_FUNCTION
State& operator-=(State &a, State b) {
  for (int i=0; i < Nfields; ++i)
    a[i] -= b[i];
  return a;
}

KOKKOS_INLINE_FUNCTION
State operator*(const State &a, real_t q) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]*q;
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator/(const State &a, real_t q) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]/q;
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator/(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]/b[i];
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator*(real_t q, const State &a) {
  return a*q;
}

KOKKOS_INLINE_FUNCTION
State operator*(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]*b[i];
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator+(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]+b[i];
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator-(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]-b[i];
  return res;
}

KOKKOS_INLINE_FUNCTION
State swap_component(State &q, IDir dir) {
  if (dir == IX)
    return q;
  else
    return {q[IR], q[IV], q[IU], q[IP]};
}


}
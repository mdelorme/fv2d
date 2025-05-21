#pragma once

namespace fv2d {

KOKKOS_INLINE_FUNCTION
State getStateFromArray(Array arr, int i, int j) {
  return {arr(j, i, IR),
          arr(j, i, IU),
          arr(j, i, IV),
          arr(j, i, IW),
          arr(j, i, IP),
          arr(j, i, IBX),
          arr(j, i, IBY),
          arr(j, i, IBZ),
          arr(j, i, IPSI)};
} 

KOKKOS_INLINE_FUNCTION
void setStateInArray(Array arr, int i, int j, State st) {
  for (int ivar=0; ivar < Nfields; ++ivar)
    arr(j, i, ivar) = st[ivar];
}

KOKKOS_INLINE_FUNCTION
State primToCons(State &q, const DeviceParams &params) {
  State res;
  res[IR] = q[IR];
  res[IU] = q[IR]*q[IU];
  res[IV] = q[IR]*q[IV];
  res[IW] = q[IR]*q[IW];

  real_t Ek = 0.5 * q[IR] * (q[IU]*q[IU] + q[IV]*q[IV] + q[IW]*q[IW]);
  real_t Em = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);
  res[IE] = Ek + Em + q[IP] / (params.gamma0-1.0);
  res[IBX] = q[IBX];
  res[IBY] = q[IBY];
  res[IBZ] = q[IBZ];
  res[IPSI] = q[IPSI];
  return res;
}


KOKKOS_INLINE_FUNCTION
State consToPrim(State &u, const DeviceParams &params) {
  State res;
  res[IR] = u[IR];
  res[IU] = u[IU] / u[IR];
  res[IV] = u[IV] / u[IR];
  res[IW] = u[IW] / u[IR];
  

  real_t Ek = 0.5 * res[IR] * (res[IU]*res[IU] + res[IV]*res[IV] + res[IW]*res[IW]);
  real_t Em = 0.5 * (u[IBX]*u[IBX] + u[IBY]*u[IBY] + u[IBZ]*u[IBZ]);
  res[IP] = (u[IE] - Ek - Em) * (params.gamma0-1.0);
  res[IBX] = u[IBX];
  res[IBY] = u[IBY];
  res[IBZ] = u[IBZ];
  res[IPSI] = u[IPSI];
  return res; 
}

KOKKOS_INLINE_FUNCTION
real_t speedOfSound(State &q, const DeviceParams &params) {
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
State operator*(real_t q, const State &a) {
  return a*q;
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
    return {q[IR], q[IV], q[IU], q[IW], q[IP], q[IBY], q[IBX], q[IBZ], q[IPSI]};
}
}

#pragma once

namespace fv2d {

KOKKOS_INLINE_FUNCTION
State getStateFromArray(Array arr, int i, int j) {
  return {arr(j, i, IR),
          arr(j, i, IU),
          arr(j, i, IV),
          arr(j, i, IP),
          arr(j, i, IBX),
          arr(j, i, IBY),
          arr(j, i, IBZ)};
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
  real_t Em = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);
  res[IE] = (Ek + Em + q[IP] / (params.gamma0-1.0));
  res[IBX] = q[IBX];
  res[IBY] = q[IBY];
  res[IBZ] = q[IBZ];
  return res;
}


KOKKOS_INLINE_FUNCTION
State consToPrim(State &u, const Params &params) {
  State res;
  res[IR] = u[IR];
  res[IU] = u[IU] / u[IR];
  res[IV] = u[IV] / u[IR];

  real_t Ek = 0.5 * res[IR] * (res[IU]*res[IU] + res[IV]*res[IV]);
  real_t Em = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);
  res[IP] = (u[IE] - Ek - Em) * (params.gamma0-1.0);
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
    return {q[IR], q[IV], q[IU], q[IP], q[IBY], q[IBX], q[IBZ]};
}

/** TODO Lucas OK*/
KOKKOS_INLINE_FUNCTION
State computeFlux(State &q, const real_t E, Params &params) {
  // const real_t Ek = 0.5 * q[IR] * (q[IU] * q[IU] + q[IV] * q[IV]);
  const real_t Em = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);
  // const real_t E = (q[IP] / (params.gamma0-1.0) + Ek + Em);
  const real_t Ptot = q[IP] + Em;

  State fout {
    q[IR] * q[IU],
    q[IR] * q[IU] * q[IU] + Ptot - q[IBX] * q[IBX],
    q[IR] * q[IV] * q[IU] - q[IBY] * q[IBX],
    q[IR] * q[IW] * q[IU] - q[IBZ] * q[IBX],
    q[IBY] * q[IU] - q[IBX] * q[IV],
    q[IBZ] * q[IU] - q[IBX] * q[IW],
    (E + Ptot) * q[IU] - q[IBX] * (q[IU]*q[IBX] + q[IV]*q[IBY] + q[IW]*q[IBZ])

  };

  return fout;
}

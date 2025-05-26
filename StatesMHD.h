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
  real_t Epsi = (params.riemann_solver==IDEALGLM ? 0.5*q[IPSI]*q[IPSI] : 0.0);
  res[IE] = Ek + Em + Epsi + q[IP] / (params.gamma0-1.0);
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
  real_t Epsi = (params.riemann_solver==IDEALGLM ? 0.5*u[IPSI]*u[IPSI] : 0.0);
 
  res[IP] = (u[IE] - Ek - Em - Epsi) * (params.gamma0-1.0);
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
real_t fastMagnetoAcousticSpeed(State &q, const DeviceParams &params, IDir idir) {
  auto iB = IBX + static_cast<IVar>(idir); // IBX, IBY, or IBZ depending on idir
  const real_t cs = speedOfSound(q, params);
  const real_t c02  = cs*cs;
  const real_t B2 = q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ];
  const real_t ca2  = B2 / q[IR];
  const real_t cap2 = q[iB]*q[iB]/q[IR];
  return sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.0*c02*cap2));
}

KOKKOS_INLINE_FUNCTION
real_t ComputeGlobalDivergenceSpeed(Array Q, const Params &full_params) {
  auto params = full_params.device_params;
  real_t u_max = 0.0;
  real_t lambda_max = 0.0;
  Kokkos::parallel_reduce("Compute Global Divergece Speed",
                          full_params.range_dom,
                          KOKKOS_LAMBDA(int i, int j, real_t& u_max, real_t& lamba_max) {
                              State q = getStateFromArray(Q, i, j);
                              real_t umax_loc = Kokkos::max({Kokkos::abs(q[IU]), Kokkos::abs(q[IV]), Kokkos::abs(q[IW])});
                              real_t lambda_x = Kokkos::max(Kokkos::abs(q[IU] - fastMagnetoAcousticSpeed(q, params, IX)), Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX)));
                              real_t lambda_y = Kokkos::max(Kokkos::abs(q[IV] - fastMagnetoAcousticSpeed(q, params, IY)), Kokkos::abs(q[IV] + fastMagnetoAcousticSpeed(q, params, IY)));
                              real_t lambda_loc = Kokkos::max(lambda_x, lambda_y);
                              u_max = Kokkos::max(u_max, umax_loc);
                              lamba_max = Kokkos::max(lamba_max, lambda_loc);
                          },
                          Kokkos::Max<real_t>(u_max),
                          Kokkos::Max<real_t>(lambda_max));
  return lambda_max - u_max;
};

KOKKOS_INLINE_FUNCTION
real_t logMean(const real_t xl, const real_t xr, const real_t epsilon = 1e-2){
    const real_t zeta = xl/xr;
    const real_t f = (zeta - 1.0) / (zeta + 1.0);
    const real_t u = f * f;
    real_t F;
    if (u < epsilon){
      F = 1.0 + u/3.0 + u*u/5.0 + u*u*u/7.0;
    }
    else{
      F = Kokkos::log(zeta) / 2.0 / f;
    }
    return (xr + xl) / (2 * F);
  };

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
State operator*(const State &a, const State &b) {
  State res;
  for (int i=0; i < Nfields; ++i)
    res[i] = a[i]*b[i];
  return res;
}

KOKKOS_INLINE_FUNCTION
State operator*=(State &a, real_t q) {
  for (int i=0; i < Nfields; ++i)
    a[i] *= q;
  return a;
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

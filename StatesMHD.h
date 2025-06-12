#pragma once

#include "LinAlg.h"

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
  return Kokkos::sqrt(q[IP] * params.gamma0 / q[IR]);
}

KOKKOS_INLINE_FUNCTION
real_t fastMagnetoAcousticSpeed(State &q, const DeviceParams &params, IDir idir) {
  const int iB = IBX + idir; //static_cast<IVar>(idir); // IBX, IBY, or IBZ depending on idir
  const real_t cs = speedOfSound(q, params);
  const real_t c02  = cs*cs;
  const real_t B2 = q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ];
  const real_t ca2  = B2 / q[IR];
  const real_t cap2 = q[iB]*q[iB]/q[IR];
  return Kokkos::sqrt(0.5*(c02+ca2)+0.5*Kokkos::sqrt((c02+ca2)*(c02+ca2)-4.0*c02*cap2));
}

real_t ComputeGlobalDivergenceSpeed(Array Q, const Params &full_params) {
  auto params = full_params.device_params;
  real_t u_max = 0.0;
  real_t lambda_max = 0.0;
  Kokkos::parallel_reduce("Compute Global Divergece Speed",
    full_params.range_dom,
    KOKKOS_LAMBDA(int i, int j, real_t& u_max, real_t& lambda_max) {
      State q = getStateFromArray(Q, i, j);
      real_t umax_loc = Kokkos::max({Kokkos::abs(q[IU]), Kokkos::abs(q[IV]), Kokkos::abs(q[IW])});
      real_t lambda_x = Kokkos::max(Kokkos::abs(q[IU] - fastMagnetoAcousticSpeed(q, params, IX)), Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX)));
      real_t lambda_y = Kokkos::max(Kokkos::abs(q[IV] - fastMagnetoAcousticSpeed(q, params, IY)), Kokkos::abs(q[IV] + fastMagnetoAcousticSpeed(q, params, IY)));
      real_t lambda_loc = Kokkos::max(lambda_x, lambda_y);
      u_max = Kokkos::max(u_max, umax_loc);
      lambda_max = Kokkos::max(lambda_max, lambda_loc);
    },
    Kokkos::Max<real_t>(u_max),
    Kokkos::Max<real_t>(lambda_max));
    return lambda_max - u_max;
  }
  
real_t ComputeLambdaMax(Array Q, const Params &full_params) {
  auto params = full_params.device_params;
  real_t lambda_max = 0.0;
  Kokkos::parallel_reduce("Compute Lambda Max",
    full_params.range_dom,
    KOKKOS_LAMBDA(int i, int j, real_t& lamba_max) {
      State q = getStateFromArray(Q, i, j);
      real_t lambda_x = Kokkos::max(Kokkos::abs(q[IU] - fastMagnetoAcousticSpeed(q, params, IX)), Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX)));
      real_t lambda_y = Kokkos::max(Kokkos::abs(q[IV] - fastMagnetoAcousticSpeed(q, params, IY)), Kokkos::abs(q[IV] + fastMagnetoAcousticSpeed(q, params, IY)));
      real_t lambda_loc = Kokkos::max(lambda_x, lambda_y);
      lamba_max = Kokkos::max(lamba_max, lambda_loc);
    },
    Kokkos::Max<real_t>(lambda_max));
    return lambda_max;
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

KOKKOS_INLINE_FUNCTION
State getConsJumpState(const State &qL, const State &qR, const DeviceParams &params) {
  State qJump = qR - qL;
  State qMixed = 0.5 * (qR*qR - qL*qL); //{{q}}[[q]]
  const real_t rhoLn = logMean(qL[IR], qR[IR]);
  const real_t betaL = 0.5 * qL[IR]/qL[IP];
  const real_t betaR = 0.5 * qR[IR]/qR[IP];
  const real_t betaJump = betaR - betaL;
  const real_t betaAvg = 0.5 * (qL[IR]+qR[IR])/(qL[IP]+qR[IP]);
  const real_t betaLn = 0.5 * rhoLn / logMean(qL[IP], qR[IP]);
  const real_t beta2bar = 2.0*betaAvg*betaAvg - 0.5*(betaL*betaL + betaR*betaR);
  const real_t u2bar = qL[IU]*qR[IU] + qL[IV]*qR[IV] + qL[IW]*qR[IW];
  real_t E = qJump[IR]/(2.0*(params.gamma0-1.0)*betaLn) + 0.5*u2bar*qJump[IR] + 0.5*(qL[IR]+qR[IR])*(qMixed[IU] + qMixed[IV] + qMixed[IW]) - 0.25*(qL[IR]+qR[IR])/((params.gamma0-1.0)*beta2bar)*betaJump + qMixed[IBX] + qMixed[IBY] + qMixed[IBZ] + qMixed[IPSI];
  return {
    qJump[IR], 
    qR[IR]*qR[IU]-qL[IR]*qL[IU], 
    qR[IR]*qR[IV]-qL[IR]*qL[IV],
    qR[IR]*qR[IW]-qL[IR]*qL[IW],
    E,
    qJump[IBX],
    qJump[IBY],
    qJump[IBZ],
    qJump[IPSI]
  };
}

KOKKOS_INLINE_FUNCTION
State getEntropyJumpState(State &qL, State &qR, const DeviceParams &params) {
  const State qJump = qR - qL;
  const State qAvg = 0.5 * (qL + qR);
  const real_t rhoLn = logMean(qL[IR], qR[IR]);
  const real_t betaL = 0.5 * qL[IR]/qL[IP]; 
  const real_t betaR = 0.5 * qR[IR]/qR[IP];
  const real_t betaJump = betaR - betaL;
  const real_t betaAvg = 0.5 * qAvg[IR]/qAvg[IP];
  const real_t betaLn = 0.5 * rhoLn / logMean(qL[IP], qR[IP]);
  const real_t v2Avg = 0.5 * (qL[IU]*qL[IU]+qR[IU]*qR[IU] + qL[IV]*qL[IV]+qR[IV]*qR[IV] + qL[IW]*qL[IW]+qR[IW]*qR[IW]);
  return {
    qJump[IR]/rhoLn + betaJump/(betaLn*(params.gamma0-1.0)) - v2Avg*betaJump - 2.0*betaAvg*(qAvg[IU]*qJump[IU] + qAvg[IV]*qJump[IV] + qAvg[IW]*qJump[IW]),
    2.0 * (betaAvg * qJump[IU] + qAvg[IU]*betaJump),
    2.0 * (betaAvg * qJump[IV] + qAvg[IV]*betaJump),
    2.0 * (betaAvg * qJump[IW] + qAvg[IW]*betaJump),
    -2.0 * betaJump,
    2.0 * (betaAvg * qJump[IBX] + qAvg[IBX]*betaJump),
    2.0 * (betaAvg * qJump[IBY] + qAvg[IBY]*betaJump),
    2.0 * (betaAvg * qJump[IBZ] + qAvg[IBZ]*betaJump),
    2.0 * (betaAvg * qJump[IPSI] + qAvg[IPSI]*betaJump)
  };
}
}

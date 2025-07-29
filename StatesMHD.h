#pragma once

#include "LinAlg.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
State getStateFromArray(Array arr, int i, int j) {
  State res;
  res[IR] = arr(j, i, IR);
  res[IU] = arr(j, i, IU);
  res[IV] = arr(j, i, IV);
  res[IW] = arr(j, i, IW);
  res[IP] = arr(j, i, IP);
  res[IBX] = arr(j, i, IBX);
  res[IBY] = arr(j, i, IBY);
  res[IBZ] = arr(j, i, IBZ);
  res[IPSI] = arr(j, i, IPSI);
  return res;
} 

KOKKOS_INLINE_FUNCTION
void setStateInArray(Array arr, int i, int j, State st) {
  // for (int ivar=0; ivar < Nfields; ++ivar)
  // arr(j, i, ivar) = st[ivar];
  arr(j, i, IR) = st[IR];
  arr(j, i, IU) = st[IU];
  arr(j, i, IV) = st[IV];
  arr(j, i, IW) = st[IW];
  arr(j, i, IP) = st[IP];
  arr(j, i, IBX) = st[IBX];
  arr(j, i, IBY) = st[IBY];
  arr(j, i, IBZ) = st[IBZ];
  arr(j, i, IPSI) = st[IPSI];
}

KOKKOS_INLINE_FUNCTION
State zero_state() {
  State res {};
  for (int ivar=0; ivar < Nfields; ++ivar) {
    res[ivar] = 0.0;
  }
  return res;
}

KOKKOS_INLINE_FUNCTION
State primToCons(State &q, const DeviceParams &params) {
  State res;
  res[IR] = q[IR];
  res[IU] = q[IR]*q[IU];
  res[IV] = q[IR]*q[IV];
  res[IW] = q[IR]*q[IW];
  
  real_t Ek = 0.5 * (res[IU]*res[IU] + res[IV]*res[IV] + res[IW]*res[IW]) / q[IR];
  real_t Em = 0.5 * (q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]);
  real_t Epsi = (params.riemann_solver==IDEALGLM ? 0.5*q[IPSI]*q[IPSI] : 0.0);
  res[IE] = (Ek + q[IP] / (params.gamma0-1.0) + Em + Epsi);
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
  const real_t a2 = params.gamma0 * q[IP] / q[IR];
  const real_t sqrt_rho = Kokkos::sqrt(q[IR]);
  const real_t b1 = q[IBX] / sqrt_rho;
  const real_t b2 = q[IBY] / sqrt_rho;
  const real_t b3 = q[IBZ] / sqrt_rho;
  const real_t B2 = b1*b1 + b2*b2 + b3*b3;
  const real_t bi = (idir == IX ? b1 : b2);

  return Kokkos::sqrt(
    0.5 * (a2 + B2 + Kokkos::sqrt((a2 + B2)*(a2 + B2) - 4.0 * a2 * bi*bi))
  );
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
    else {
      State qs;

      qs[IR] = q[IR];
      qs[IU] = q[IV];
      qs[IV] = q[IU];
      qs[IW] = q[IW];
      qs[IP] = q[IP];
      qs[IBX] = q[IBY]; 
      qs[IBY] = q[IBX];
      qs[IBZ] = q[IBZ];
      qs[IPSI] = q[IPSI];
      return qs;
    }
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
State primToEntropy(State &q, const DeviceParams &params) {
  State res {};
  const real_t beta = 0.5 * q[IR]/q[IP];
  const real_t s = (1 - params.gamma0)*Kokkos::log(q[IR]) - Kokkos::log(2.0*beta);
  res[IR] = (params.gamma0 - s)/(params.gamma0 - 1.0) - beta * (q[IU]*q[IU] + q[IV]*q[IV] + q[IW]*q[IW]);
  res[IU] = 2.0 * beta * q[IU];
  res[IV] = 2.0 * beta * q[IV];
  res[IW] = 2.0 * beta * q[IW];
  res[IE] = -2.0 * beta;
  res[IBX] = 2.0 * beta * q[IBX];
  res[IBY] = 2.0 * beta * q[IBY];
  res[IBZ] = 2.0 * beta * q[IBZ];
  res[IPSI] = 2.0 * beta * q[IPSI];
  return res;
}


KOKKOS_INLINE_FUNCTION
State getEntropyJumpState(State &qL, State &qR, const DeviceParams &params) {
  State res{};
  real_t v2L = qL[IU]*qL[IU] + qL[IV]*qL[IV] + qL[IW]*qL[IW];
  real_t v2R = qR[IU]*qR[IU] + qR[IV]*qR[IV] + qR[IW]*qR[IW];
  real_t betaL = 0.5 * qL[IR]/qL[IP];
  real_t betaR = 0.5 * qR[IR]/qR[IP];
  real_t betaLn = logMean(betaL, betaR);
  res[IR] = params.gamma0 * (Kokkos::log(qR[IR]/qL[IR]) - Kokkos::log(qR[IP]/qL[IP]))/(params.gamma0 - 1.0) - (betaR*v2R - betaL*v2L);
  res[IU] = 2.0 * (betaR*qR[IU] - betaL*qL[IU]);
  res[IV] = 2.0 * (betaR*qR[IV] - betaL*qL[IV]);
  res[IW] = 2.0 * (betaR*qR[IW] - betaL*qL[IW]);
  res[IE] = -2.0 * (betaR - betaL);
  res[IBX] = 2.0 * (betaR*qR[IBX] - betaL*qL[IBX]);
  res[IBY] = 2.0 * (betaR*qR[IBY] - betaL*qL[IBY]);
  res[IBZ] = 2.0 * (betaR*qR[IBZ] - betaL*qL[IBZ]);
  res[IPSI] = 2.0 * (betaR*qR[IPSI] - betaL*qL[IPSI]);
  return res;
}
}
#pragma once

#include "SimInfo.h"
namespace fv2d {
/* 
KOKKOS_INLINE_FUNCTION
void hll_rot(State &qL, State &qR, State& flux, real_t &pout, const Params &params) {
  const real_t aL = speedOfSound(qL, params);
  const real_t aR = speedOfSound(qR, params);

  // Davis' estimates for the signal speed
  const real_t sminL = qL[IU] - aL;
  const real_t smaxL = qL[IU] + aL;
  const real_t sminR = qR[IU] - aR;
  const real_t smaxR = qR[IU] + aR;

  const real_t SL = fmin(sminL, sminR);
  const real_t SR = fmax(smaxL, smaxR);

  State FL = computeFlux(qL, params);
  State FR = computeFlux(qR, params);

  if (SL >= 0.0) {
    flux = FL;
    pout = qL[IP];
  }
  else if (SR <= 0.0) {
    flux = FR;
    pout = qR[IP];
  }
  else {
    State uL = primToCons(qL, params);
    State uR = primToCons(qR, params);
    pout = 0.5 * (qL[IP] + qR[IP]);
    flux = (SR*FL - SL*FR + SL*SR*(uR-uL)) / (SR-SL);
  } 
} */

KOKKOS_INLINE_FUNCTION
void hllc_rot(State &qL, State &qR, State &flux, real_t &pout, Pos n /*normale*/, const Params &params) {
  const real_t rL = qL[IR];
  const real_t uL = qL[IU];
  const real_t vL = qL[IV];
  const real_t pL = qL[IP];

  const real_t rR = qR[IR];
  const real_t uR = qR[IU];
  const real_t vR = qR[IV];
  const real_t pR = qR[IP];

  const real_t nx = n[IX];
  const real_t ny = n[IY];
  const real_t cL = uL * nx + vL * ny;
  const real_t cR = uR * nx + vR * ny;
  

  const real_t entho = 1.0 / (params.gamma0-1.0);

  const real_t ekL = 0.5 * rL * (uL * uL + vL * vL);
  const real_t EL  = ekL + pL * entho;
  const real_t ekR = 0.5 * rR * (uR * uR + vR * vR);
  const real_t ER  = ekR + pR * entho;

  const real_t cfastL = speedOfSound(qL, params);
  const real_t cfastR = speedOfSound(qR, params);

  real_t SL, SR;
  {
    const real_t R = sqrt(rR / rL);
    const real_t ut = (uL + uR * R) / (1.0 + R);
    const real_t vt = (vL + vR * R) / (1.0 + R);
    const real_t ct = ut * nx + vt * ny;
    const real_t HL = (EL + pL) / rL; // ekL ?
    const real_t HR = (ER + pR) / rR; // ekR ?
    const real_t Ht = (HL + HR * R) / (1.0 + R);
    const real_t cfastt = sqrt((params.gamma0-1.0) * (Ht - 0.5 * (ut*ut + vt*vt)));
    SL = fmin(cL - cfastL, ct - cfastt);
    SR = fmax(cR + cfastR, ct + cfastt);
  }

  const real_t rcL = rL*(cL-SL);
  const real_t rcR = rR*(SR-cR);

  const real_t SM = (rcR*cR + rcL*cL + (pL-pR))/(rcR+rcL);
  const real_t pS = (rcR*pL+rcL*pR+rcL*rcR*(cL-cR))/(rcR+rcL);
  // const real_t pS = pL + rcL * (cL - SM);

  const real_t rSL = rL*(SL-cL)/(SL-SM);
  const real_t ESL = ((SL-cL)*EL-pL*cL+pS*SM)/(SL-SM);

  const real_t rSR = rR*(SR-cR)/(SR-SM);
  const real_t ESR = ((SR-cR)*ER-pR*cR+pS*SM)/(SR-SM);

  State st;
  real_t E;
  real_t c; // advection speed in normale direction
  if (SL > 0.0) {
    st = qL;
    c = cL;
    E = EL;
    pout = pL;
  }
  else if (SM > 0.0) {
    st[IR] = rSL;
    st[IU] = (-rcL * uL + (pS - pL) * nx) / (SL - SM);
    st[IV] = (-rcL * vL + (pS - pL) * ny) / (SL - SM);
    st[IP] = pS;
    c = SM; // ct ? 
    E = ESL;
    pout = pS;
  }
  else if (SR > 0.0) {
    st[IR] = rSR;
    st[IU] = (rcR * uR + (pS - pR) * nx) / (SR - SM);
    st[IV] = (rcR * vR + (pS - pR) * ny) / (SR - SM);
    st[IP] = pS;
    c = SM; // ct ?
    E = ESR;
    pout = pS;
  }
  else {
    st = qR;
    c = cR;
    E = ER;
    pout = pR;
  }

  flux[IR] = st[IR]*c;
  flux[IU] = st[IU]*c + st[IP]*nx;
  flux[IV] = st[IV]*c + st[IP]*ny;
  flux[IE] = (E + st[IP])*c;
}

}
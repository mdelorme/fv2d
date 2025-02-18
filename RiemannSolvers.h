#pragma once

namespace fv2d {

KOKKOS_INLINE_FUNCTION
void hll(State &qL, State &qR, State& flux, real_t &pout, const DeviceParams &params) {
  const real_t aL = speedOfSound(qL, params);
  const real_t aR = speedOfSound(qR, params);

  // Davis' estimates for the signal speed
  const real_t sminL = qL[IU] - aL;
  const real_t smaxL = qL[IU] + aL;
  const real_t sminR = qR[IU] - aR;
  const real_t smaxR = qR[IU] + aR;

  const real_t SL = fmin(sminL, sminR);
  const real_t SR = fmax(smaxL, smaxR);


  auto computeFlux = [&](State &q, const DeviceParams &params) {
    const real_t Ek = 0.5 * q[IR] * (q[IU] * q[IU] + q[IV] * q[IV]);
    const real_t E = (q[IP] / (params.gamma0-1.0) + Ek);

    State fout {
      q[IR]*q[IU],
      q[IR]*q[IU]*q[IU] + q[IP],
      q[IR]*q[IU]*q[IV],
      (q[IP] + E) * q[IU]
    };

    return fout;
  };

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
}

KOKKOS_INLINE_FUNCTION
void hllc(State &qL, State &qR, State &flux, real_t &pout, const DeviceParams &params) {
  const real_t rL = qL[IR];
  const real_t uL = qL[IU];
  const real_t vL = qL[IV];
  const real_t pL = qL[IP];

  const real_t rR = qR[IR];
  const real_t uR = qR[IU];
  const real_t vR = qR[IV];
  const real_t pR = qR[IP];

  const real_t entho = 1.0 / (params.gamma0-1.0);

  const real_t ekL = 0.5 * rL * (uL * uL + vL * vL);
  const real_t EL  = ekL + pL * entho;
  const real_t ekR = 0.5 * rR * (uR * uR + vR * vR);
  const real_t ER  = ekR + pR * entho;

  const real_t cfastL = speedOfSound(qL, params);
  const real_t cfastR = speedOfSound(qR, params);

  const real_t SL = fmin(uL, uR) - fmax(cfastL, cfastR);
  const real_t SR = fmax(uL, uR) + fmax(cfastL, cfastR);


  const real_t rcL = rL*(uL-SL);
  const real_t rcR = rR*(SR-uR);

  const real_t uS = (rcR*uR + rcL*uL + (pL-pR))/(rcR+rcL);
  const real_t pS = (rcR*pL+rcL*pR+rcL*rcR*(uL-uR))/(rcR+rcL);

  const real_t rSL = rL*(SL-uL)/(SL-uS);
  const real_t ESL = ((SL-uL)*EL-pL*uL+pS*uS)/(SL-uS);

  const real_t rSR = rR*(SR-uR)/(SR-uS);
  const real_t ESR = ((SR-uR)*ER-pR*uR+pS*uS)/(SR-uS);

  State st;
  real_t E;
  if (SL > 0.0) {
    st = qL;
    E = EL;
    pout = pL;
  }
  else if (uS > 0.0) {
    st[IR] = rSL;
    st[IU] = uS;
    st[IV] = qL[IV];
    st[IP] = pS;
    E = ESL;
    pout = pS;
  }
  else if (SR > 0.0) {
    st[IR] = rSR;
    st[IU] = uS;
    st[IV] = qR[IV];
    st[IP] = pS;
    E = ESR;
    pout = pS;
  }
  else {
    st = qR;
    E = ER;
    pout = pR;
  }

  flux[IR] = st[IR]*st[IU];
  flux[IU] = st[IR]*st[IU]*st[IU]+st[IP];
  flux[IV] = flux[IR]*st[IV];
  flux[IE] = (E + st[IP])*st[IU];
}

/**
 * @brief Flux Splitting Lagrange Projection Solver
 * 
 * FSLP solver as defined in Bourgeois et al. 2024 : "Recasting an operator splitting solver into a 
 * standard finite volume flux-based algorithm. The case of a Lagrange-projection-type method for gas 
 * dynamics", Journal of Computational Physics, vol 496.
 */
KOKKOS_INLINE_FUNCTION
void fslp(State &qL, State &qR, State &flux, real_t &pout, real_t gdx, const DeviceParams &params) {
  // 1. Basic quantities
  const real_t rhoL = qL[IR];
  const real_t uL   = qL[IU];
  const real_t vL   = qL[IV];
  const real_t pL   = qL[IP];
  const real_t csL = sqrt(params.gamma0 * pL / rhoL);

  const real_t rhoR = qR[IR];
  const real_t uR   = qR[IU];
  const real_t vR   = qR[IV];
  const real_t pR   = qR[IP];
  const real_t csR = sqrt(params.gamma0 * pR / rhoR);

  // 2. Calculating theta and a
  const real_t ai = params.fslp_K * Kokkos::max(rhoL*csL, rhoR*csR);           // eq (17)
  const real_t theta = Kokkos::min(1.0, Kokkos::max(Kokkos::abs(uL)/csL, Kokkos::abs(uR)/csR));  // eq (77)

  // 3. Calculating u* and PI*
  //                                                  vvv this minus comes from g = -grad phi
  const real_t ustar = 0.5*(uR+uL) - 0.5 / ai * (pR-pL - 0.5*(rhoL+rhoR)*gdx);  // eq (15)
  const real_t Pi    = 0.5*(pR+pL) - theta * 0.5*ai*(uR-uL);                    // eq (15)

  // 4. Upwinding
  const State& qstar = (ustar > 0 ? qL : qR);                                         // eq (32)
  const real_t Ekstar = 0.5 * qstar[IR] * (qstar[IU]*qstar[IU]+qstar[IV]*qstar[IV]); 
  const real_t Estar = Ekstar + qstar[IP]/(params.gamma0-1.0);

  // 5. Calculating flux : eq (31)
  flux[IR] = ustar*qstar[IR];
  flux[IU] = ustar*qstar[IR]*qstar[IU] + Pi;
  flux[IV] = ustar*qstar[IR]*qstar[IV];
  flux[IE] = ustar*(Estar + Pi);

  //printf("ai=%lf; theta=%lf; csL=%lf; csR=%lf; ustar=%lf; Pi=%lf\n", ai, theta, csL, csR, ustar, Pi);
}

}
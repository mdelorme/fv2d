#pragma once

namespace fv2d {

KOKKOS_INLINE_FUNCTION
void hll(State &qL, State &qR, State& flux, real_t &pout, const Params &params) {
  const real_t aL = speedOfSound(qL, params);
  const real_t aR = speedOfSound(qR, params);

  // Davis' estimates for the signal speed
  const real_t sminL = qL[IU] - aL;
  const real_t smaxL = qL[IU] + aL;
  const real_t sminR = qR[IU] - aR;
  const real_t smaxR = qR[IU] + aR;

  const real_t SL = fmin(sminL, sminR);
  const real_t SR = fmax(smaxL, smaxR);


  auto computeFlux = [&](State &q, const Params &params) {
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
void hllc(State &qL, State &qR, State &flux, real_t &pout, const Params &params) {
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

/** TODO Lucas OK */
#ifdef MHD

KOKKOS_INLINE_FUNCTION
void hlld(State &qL, State &qR, State &flux, real_t &p_gas_out, const real_t Bx, const Params &params) {
  const real_t Bsgn = (Bx < 0.0 ? -1.0 : 1.0);
  const real_t smalle = std::pow(10.0, -5.0);
  // Deref of states
  const real_t uL = qL[IU];
  const real_t vL = qL[IV];
  const real_t wL = qL[IW];
  const real_t pL = qL[IP];
  const real_t rL = qL[IR];
  const real_t ByL = qL[IBY];
  const real_t BzL = qL[IBZ];
  const real_t B2L = Bx*Bx+ByL*ByL+BzL*BzL;
  const real_t pTL = pL + 0.5 * B2L;
  const real_t EL  = pL / (params.gamma0-1.0) + 0.5*rL*(uL*uL+vL*vL+wL*wL) + 0.5*B2L;

  const real_t uR = qR[IU];
  const real_t vR = qR[IV];
  const real_t wR = qR[IW];
  const real_t pR = qR[IP];
  const real_t rR = qR[IR];
  const real_t ByR = qR[IBY];
  const real_t BzR = qR[IBZ];
  const real_t B2R = Bx*Bx+ByR*ByR+BzR*BzR;
  const real_t pTR = pR + 0.5 * B2R;
  const real_t ER  = pR / (params.gamma0-1.0) + 0.5*rR*(uR*uR+vR*vR+wR*wR) + 0.5*B2R;

  auto computeFastMagnetoAcousticSpeed = [&](const State &q, const Params &params) {
    const real_t gp = params.gamma0 * q[IP];
    const real_t B2 = Bx*Bx + q[IBY]*q[IBY] + q[IBZ]*q[IBZ];
    
    return sqrt(0.5 * (gp + B2 + sqrt((gp + B2)*(gp + B2) - 4.0*gp*Bx*Bx)) / q[IR]);
  };

  const real_t cfL = computeFastMagnetoAcousticSpeed(qL, params);
  const real_t cfR = computeFastMagnetoAcousticSpeed(qR, params);
  
  // HLL Wave speed
  const real_t SL = fmin(uL, uR) - fmax(cfL, cfR);
  const real_t SR = fmax(uL, uR) + fmax(cfL, cfR);

  // Lagrangian speed of sound
  const real_t rCL = rL*(uL-SL);
  const real_t rCR = rR*(SR-uR);

  // Entropy wave speed
  const real_t uS = (rCR*uR + rCL*uL - pTR + pTL) / (rCR+rCL);
  
  // Single Star state
  const real_t pTS = (rCR*pTL + rCL*pTR - rCR*rCL*(uR-uL)) / (rCR+rCL); 

  // Single star densities
  const real_t rLS = rL * (SL-uL)/(SL-uS);
  const real_t rRS = rR * (SR-uR)/(SR-uS);

  // Single star velocities
  const real_t econvL = rL*(SL-uL)*(SL-uS)-Bx*Bx;
  const real_t econvR = rR*(SR-uR)*(SR-uS)-Bx*Bx;

  const real_t uconvL = (uS-uL) / econvL;
  const real_t uconvR = (uS-uR) / econvR;
  const real_t BconvL = (rCL*rCL/rL - Bx*Bx) / econvL;
  const real_t BconvR = (rCR*rCR/rR - Bx*Bx) / econvR;

  real_t vLS, vRS, wLS, wRS, ByLS, ByRS, BzLS, BzRS;

  // Switching to two state on the left ?
  if (std::abs(econvL) < smalle*Bx*Bx) {
    vLS = vL;
    wLS = wL;
    ByLS = ByL;
    BzLS = BzL;
  }
  else {
    vLS = vL - Bx*ByL * uconvL;
    wLS = wL - Bx*BzL * uconvL;
    ByLS = ByL * BconvL;
    BzLS = BzL * BconvL;
  }

  // Switching to two state on the right ?
  if (std::abs(econvR) < smalle*Bx*Bx) {
    vRS  = vR;
    wRS  = wR;
    ByRS = ByR;
    BzRS = BzR;
  }
  else {
    vRS = vR - Bx*ByR * uconvR;
    wRS = wR - Bx*BzR * uconvR;
    ByRS = ByR * BconvR;
    BzRS = BzR * BconvR;
  }

  // Single star total energy
  const real_t udotBL = uL*Bx+vL*ByL+wL*BzL;
  const real_t udotBR = uR*Bx+vR*ByR+wR*BzR;
  const real_t uSdotBSL = uS*Bx+vLS*ByLS+wLS*BzLS;
  const real_t uSdotBSR = uS*Bx+vRS*ByRS+wRS*BzRS;

  const real_t ELS = ((SL-uL)*EL - pTL*uL + pTS*uS + Bx*(udotBL - uSdotBSL)) / (SL-uS);
  const real_t ERS = ((SR-uR)*ER - pTR*uR + pTS*uS + Bx*(udotBR - uSdotBSR)) / (SR-uS);

  // Alfven wave speeds
  const real_t srLS = sqrt(rLS);
  const real_t srRS = sqrt(rRS);
  const real_t SLS = uS - fabs(Bx) / srLS;
  const real_t SRS = uS + fabs(Bx) / srRS;

  // Double Star state
  const real_t den_fac = 1.0 / (srLS + srRS);
  const real_t vSS = (srLS*vLS + srRS*vRS + (ByRS-ByLS)*Bsgn) * den_fac;
  const real_t wSS = (srLS*wLS + srRS*wRS + (BzRS-BzLS)*Bsgn) * den_fac;
  const real_t BySS = (srLS*ByRS + srRS*ByLS + srLS*srRS*(vRS-vLS)*Bsgn) * den_fac;
  const real_t BzSS = (srLS*BzRS + srRS*BzLS + srLS*srRS*(wRS-wLS)*Bsgn) * den_fac; 

  const real_t uSSdotBSS = uS*Bx + vSS*BySS + wSS*BzSS;

  const real_t ELSS = ELS - srLS * (uSdotBSL - uSSdotBSS) * Bsgn;
  const real_t ERSS = ERS + srRS * (uSdotBSR - uSSdotBSS) * Bsgn;

  // Lambda to compute a flux from a primitive state
  auto computeFlux = [&](const State &q, const real_t e_tot) -> State {
    State res{};

    res[IR]  = q[IR] * q[IU];
    res[IU]  = q[IR] * q[IU] * q[IU] + q[IP] - q[IBX]*q[IBX];
    res[IV]  = q[IR] * q[IU] * q[IV] - q[IBX]*q[IBY];
    res[IW]  = q[IR] * q[IU] * q[IW] - q[IBX]*q[IBZ];
    res[IBX] = 0.0;
    res[IBY] = q[IBY]*q[IU] - q[IBX]*q[IV];
    res[IBZ] = q[IBZ]*q[IU] - q[IBX]*q[IW];
    res[IE]  = (e_tot + q[IP]) * q[IU] - q[IBX]*(q[IBX]*q[IU]+q[IBY]*q[IV]+q[IBZ]*q[IW]);
    res[IPSI] = 0.0;
    return res;
  };

  // Disjunction of cases
  
  State q;
  real_t e_tot;
  if (SL > 0.0) { // qL
    q = qL;
    e_tot = EL;
    q[IP] = pTL;

    p_gas_out = qL[IP];
  }
  else if (SLS > 0.0) { // qL*
    q[IR] = rLS;
    q[IU]   = uS;
    q[IV]   = vLS;
    q[IW]   = wLS;
    q[IBX]  = Bx;
    q[IBY]  = ByLS;
    q[IBZ]  = BzLS;

    q[IP] = pTS;
    e_tot = ELS;

    p_gas_out = qL[IP];
  }
  else if (uS > 0.0) { // qL**
    q[IR] = rLS;
    q[IU]   = uS;
    q[IV]   = vSS;
    q[IW]   = wSS;
    q[IBX]  = Bx;
    q[IBY]  = BySS;
    q[IBZ]  = BzSS;

    q[IP]   = pTS;
    e_tot = ELSS;

    p_gas_out = qL[IP];
  }
  else if (SRS > 0.0) { // qR**
    q[IR] = rRS;
    q[IU]   = uS;
    q[IV]   = vSS;
    q[IW]   = wSS;
    q[IBX]  = Bx;
    q[IBY]  = BySS;
    q[IBZ]  = BzSS;

    q[IP]   = pTS;
    e_tot = ERSS;

    p_gas_out = qR[IP];
  }
  else if (SR > 0.0) { // qR*
    q[IR] = rRS;
    q[IU]   = uS;
    q[IV]   = vRS;
    q[IW]   = wRS;
    q[IBX]  = Bx;
    q[IBY]  = ByRS;
    q[IBZ]  = BzRS;

    q[IP] = pTS;
    e_tot = ERS;

    p_gas_out = qR[IP];
  }
  else { // SR < 0.0; qR
    q = qR;
    e_tot = ER;
    q[IP] = pTR;

    p_gas_out = qR[IP];
  }

  flux = computeFlux(q, e_tot);
}

void FiveWaves(State &qL, State &qR, State &flux, real_t &pout, const Params &params) {
  const uint IZ = 2;
  using vec_t = Kokkos::Array<real_t, 3>;
  constexpr real_t epsilon = 1.0e-16;
  const real_t B2L = qL[IBX]*qL[IBX] + qL[IBY]*qL[IBY] + qL[IBZ]*qL[IBZ];
  const real_t B2R = qR[IBX]*qR[IBX] + qR[IBY]*qR[IBY] + qR[IBZ]*qR[IBZ];
  const vec_t pL {
    -qL[IBX]*qL[IBX] + qL[IP] + B2L/2,
    -qL[IBX]*qL[IBY],
    -qL[IBX]*qL[IBZ]
  };

  const vec_t pR {
    -qR[IBX]*qR[IBX] + qR[IP] + B2R/2,
    -qR[IBX]*qR[IBY],
    -qR[IBX]*qR[IBZ]
  };

  // 1. Compute speeds
  const real_t csL = speedOfSound(qL, params); 
  const real_t csR = speedOfSound(qR, params);
  real_t caL = sqrt(qL[IR] * (qL[IBX]*qL[IBX] + B2L/2))+epsilon;
  real_t caR = sqrt(qR[IR] * (qR[IBX]*qR[IBX] + B2R/2))+epsilon;
  real_t cbL = sqrt(qL[IR] * (qL[IR]*csL*csL + qL[IBY]*qL[IBY] + qL[IBZ]*qL[IBZ] + B2L/2));
  real_t cbR = sqrt(qR[IR] * (qR[IR]*csR*csR + qR[IBY]*qR[IBY] + qR[IBZ]*qR[IBZ] + B2R/2));

  auto computeFastMagnetoAcousticSpeed = [&](const State &q, const real_t B2, const real_t cs) {
    const real_t c02  = cs*cs;
    const real_t ca2  = B2 / q[IR];
    const real_t cap2 = q[IBX]*q[IBX]/q[IR];
    return sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.0*c02*cap2));
  };
  
  // Using 3-wave if hyperbolicity is lost (from Dyablo)
  if ( qL[IBX]*qR[IBX] < -epsilon 
    || qL[IBY]*qR[IBY] < -epsilon
    || qL[IBZ]*qR[IBZ] < -epsilon) 
    {
    const real_t cL = qL[IR]  * computeFastMagnetoAcousticSpeed(qL,B2L, csL);
    const real_t cR = qR[IR] * computeFastMagnetoAcousticSpeed(qR, B2R, csR);
    const real_t c = fmax(cL, cR);
  
    caL = c;
    caR = c;
    cbL = c;
    cbR = c;
  }
  
  const vec_t cL {cbL, caL, caL};
  const vec_t cR {cbR, caR, caR};

  // // 2. Compute star zone
  const vec_t vL {qL[IU], qL[IV], qL[IW]};
  const vec_t vR {qR[IU], qR[IV], qR[IW]};

  vec_t Ustar{}, Pstar{};
  for (size_t i=0; i<3; ++i) {
    Ustar[i] = (cL[i]*vL[i] + cR[i]*vR[i] + pL[i] - pR[i])/(cL[i]+cR[i]);
    Pstar[i] = (cR[i]*pL[i] + cL[i]*pR[i] + cL[i]*cR[i]*(vL[i]-vR[i]))/(cL[i]+cR[i]);
  }

  State q{};
  real_t Bstar;
  if (Ustar[IX] > 0.0) {
    q = qL;
    Bstar = qR[IBX];
  }
  else {
    q = qR;
    Bstar = qL[IBX];
  }
  const real_t beta_min = 1.0e-3;
  const real_t beta = q[IP] / (0.5*(q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ]));
  bool is_low_beta = (beta < beta_min);
  q[IP] = fmax(q[IP], params.smallr);
  q[IR] = fmax(q[IR], params.smallp);
  State u = primToCons(q, params);
  //3. Commpute flux
  real_t uS = Ustar[IX];
  flux[IR]  = u[IR]  * uS;
  flux[IU]  = u[IU]  * uS + Pstar[IX];
  flux[IV]  = u[IV]  * uS + Pstar[IY];
  flux[IW]  = u[IW]  * uS + Pstar[IZ];
  flux[IE]  = u[IE]  * uS + Pstar[IX]*uS + Pstar[IY]*Ustar[IY] + Pstar[IZ]*Ustar[IZ];
  if (is_low_beta) {
    flux[IBX] = u[IBX] * uS - q[IBX] * Ustar[IX];
    flux[IBY] = u[IBY] * uS - q[IBX] * Ustar[IY];
    flux[IBZ] = u[IBZ] * uS - q[IBX] * Ustar[IZ];
  }
  else {
    flux[IBX] = u[IBX] * uS - Bstar * Ustar[IX];
    flux[IBY] = u[IBY] * uS - Bstar * Ustar[IY];
    flux[IBZ] = u[IBZ] * uS - Bstar * Ustar[IZ];
  }
  flux[IPSI] = 0.0;
  pout = Pstar[IX];
}
#endif
}

#pragma once

#include "LinAlg.h"

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

#ifdef MHD

KOKKOS_INLINE_FUNCTION
void hlld(State &qL, State &qR, State &flux, real_t &p_gas_out, const real_t Bx, const DeviceParams &params) {
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

  auto computeFastMagnetoAcousticSpeed = [&](const State &q, const DeviceParams &params) {
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

KOKKOS_INLINE_FUNCTION
void FiveWaves(State &qL, State &qR, State &flux, real_t &pout, const DeviceParams &params) {
  const uint IZ = 2;
  constexpr real_t epsilon = 1.0e-16;
  const real_t B2L = qL[IBX]*qL[IBX] + qL[IBY]*qL[IBY] + qL[IBZ]*qL[IBZ];
  const real_t B2R = qR[IBX]*qR[IBX] + qR[IBY]*qR[IBY] + qR[IBZ]*qR[IBZ];
  const Vect pL {
    -qL[IBX]*qL[IBX] + qL[IP] + B2L/2,
    -qL[IBX]*qL[IBY],
    -qL[IBX]*qL[IBZ]
  };

  const Vect pR {
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
  
  const Vect cL {cbL, caL, caL};
  const Vect cR {cbR, caR, caR};

  // // 2. Compute star zone
  const Vect vL {qL[IU], qL[IV], qL[IW]};
  const Vect vR {qR[IU], qR[IV], qR[IW]};

  Vect Ustar{}, Pstar{};
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

KOKKOS_INLINE_FUNCTION
void FluxKEPEC(State &qL, State &qR, State &flux, real_t &pout, real_t ch, const DeviceParams &params){

  const real_t betaL = 0.5 * qL[IR]/qL[IP];
  const real_t betaR = 0.5 * qR[IR]/qR[IP];
  const real_t betaAvg = 0.5 * (betaL + betaR);
  const real_t betaLn = logMean(betaL, betaR);

  const State qAvg = 0.5 * (qL + qR);
  const real_t rhoLn = logMean(qL[IR], qR[IR]);
  // sum_i {{Bi^2}}
  const real_t B2Avg = 0.5 * (qL[IBX]*qL[IBX]+qR[IBX]*qR[IBX] + 
  qL[IBY]*qL[IBY]+qR[IBY]*qR[IBY] + 
  qL[IBZ]*qL[IBZ]+qR[IBZ]*qR[IBZ]);
  //sum_i {{vi^2}}
  const real_t v2Avg = 0.5 * (qL[IU]*qL[IU] + qR[IU]*qR[IU] +
  qL[IV]*qL[IV] + qR[IV]*qR[IV] +
  qL[IW]*qL[IW] + qR[IW]*qR[IW]);
  // sum_i {{vi*Bi^2}}
  const real_t uB2Avg = 0.5 * (qL[IU]*qL[IBX]*qL[IBX] + qR[IU]*qR[IBX]*qR[IBX] +
  qL[IU]*qL[IBY]*qL[IBY] + qR[IU]*qR[IBY]*qR[IBY] +
  qL[IU]*qL[IBZ]*qL[IBZ] + qR[IU]*qR[IBZ]*qR[IBZ]);
  // sum_i {{vi*Bi}}
  const real_t uBAvg = 0.5 * (qL[IU]*qL[IBX] + qR[IU]*qR[IBX] +
  qL[IV]*qL[IBY] + qR[IV]*qR[IBY] +
  qL[IW]*qL[IBZ] + qR[IW]*qR[IBZ]);
        
  const real_t ptot = 0.5 * qAvg[IR]/betaAvg + 0.5*B2Avg;
          
  // Flux components
  flux[IR] = rhoLn * qAvg[IU];
  flux[IU] = rhoLn * qAvg[IU]*qAvg[IU] - qAvg[IBX]*qAvg[IBX] + ptot;
  flux[IV] = rhoLn * qAvg[IU]*qAvg[IV] - qAvg[IBX]*qAvg[IBY];
  flux[IW] = rhoLn * qAvg[IU]*qAvg[IW] - qAvg[IBX]*qAvg[IBZ];
  flux[IBX] = ch * qAvg[IPSI];
  flux[IBY] = qAvg[IU]*qAvg[IBY] - qAvg[IV]*qAvg[IBX];
  flux[IBZ] = qAvg[IU]*qAvg[IBZ] - qAvg[IW]*qAvg[IBX];
  flux[IPSI] = ch * qAvg[IBX];
  
  const real_t f5 = flux[IR] * (0.5/((params.gamma0-1.0)*betaLn) - 0.5*v2Avg) + 
                    flux[IU] * qAvg[IU] +
                    flux[IV] * qAvg[IV] +
                    flux[IW] * qAvg[IW] +
                    flux[IBX] * qAvg[IBX] +
                    flux[IBY] * qAvg[IBY] +
                    flux[IBZ] * qAvg[IBZ] +
                    flux[IPSI] * qAvg[IPSI] - 0.5*uB2Avg + qAvg[IBX]*uBAvg - ch * 0.5*(qL[IBX]*qL[IPSI] + qR[IBX]*qR[IPSI]);

  flux[IE] = f5;
  pout = qAvg[IP];
}

KOKKOS_INLINE_FUNCTION
State getMatrixDissipation(State &qL, State &qR, real_t ch, const DeviceParams &params) {
  const real_t betaL = 0.5 * qL[IR]/qL[IP];
  const real_t betaR = 0.5 * qR[IR]/qR[IP];
  const real_t betaAvg = 0.5 * (betaL + betaR);
  const real_t betaLn = logMean(betaL, betaR);
  const real_t rhoLn = logMean(qL[IR], qR[IR]);
  State q = 0.5 * (qL + qR); //{{q}}
  State q2 = 0.5 * (qL*qL + qR*qR); //{{q^2}}
    // Some useful constants defined in eq. (4.63)
  const real_t u2bar = qL[IU]*qR[IU] + qL[IV]*qR[IV] + qL[IW]*qR[IW];
  const real_t b1 = q[IBX]/ Kokkos::sqrt(rhoLn);
  const real_t b2 = q[IBY]/ Kokkos::sqrt(rhoLn);
  const real_t b3 = q[IBZ]/ Kokkos::sqrt(rhoLn);
  const real_t B2 = b1*b1 + b2*b2 + b3*b3;
  const real_t p_tilde = 0.5 * q[IR]/betaAvg;
  const real_t a2 = params.gamma0 * p_tilde / rhoLn;

  // Compute the discrete wave speeds
  const real_t ca = Kokkos::abs(b1);
  const real_t cf2 = 0.5 * (a2 + B2 + Kokkos::sqrt((a2 + B2)*(a2 + B2) - 4.0*a2*b1*b1));
  const real_t cf = Kokkos::sqrt(cf2);
  const real_t cs2 = 0.5 * (a2 + B2 - Kokkos::sqrt((a2 + B2)*(a2 + B2) - 4.0*a2*b1*b1));
  const real_t cs = Kokkos::sqrt(cs2);

  // Eigenvalues, as defined in eq. (4.74)
  State lambda_hat {
    Kokkos::abs(q[IU] + cf),
    Kokkos::abs(q[IU] + ca),
    Kokkos::abs(q[IU] + cs),
    Kokkos::abs(q[IU] + ch),
    Kokkos::abs(q[IU]),
    Kokkos::abs(q[IU] - ch),
    Kokkos::abs(q[IU] - cs),
    Kokkos::abs(q[IU] - ca),
    Kokkos::abs(q[IU] - cf)
  };
  // Mean State for the diagonal scaling matrix, eq. (4.70)
  const real_t z1 = 2.0 * params.gamma0 * rhoLn;
  const real_t z2 = 4.0 * betaAvg * rhoLn * rhoLn;
  const real_t z4 = 4.0 * betaAvg;
  const real_t z5 = rhoLn * (params.gamma0 - 1.0) / params.gamma0;
  State Z {1.0/z1, 1.0/z2, 1.0/z1, 1.0/z4, z5, 1.0/z4, 1.0/z1, 1.0/z2, 1.0/z1};

  // Discrete eigenvectors, eqs. (4.67)-(4.69)
  // GLM Waves: \lambda +/- \psi
  State rpsi_p {0.0, 0.0, 0.0, 0.0, q[IBX] + q[IPSI], 1.0, 0.0, 0.0, 1.0}; // r\lambda_{+ \psi}
  State rpsi_m {0.0, 0.0, 0.0, 0.0, q[IBX] - q[IPSI], 1.0, 0.0, 0.0, -1.0}; // r\lambda_{- \psi}  
  // Entropy wave : \lambda_E
  State rE {1.0, q[IU], q[IV], q[IW], 0.5*u2bar, 0.0, 0.0, 0.0, 0.0};
  // Alfvén Waves : \lambda_{\pma}
  const real_t bTrans = b2*b2 + b3*b3;
  const real_t chi2 = b2 / bTrans;
  const real_t chi3 = b3 / bTrans;
  const real_t rhoSrho = rhoLn*Kokkos::sqrt(q[IR]);
  State rA_p {0.0, 0.0, rhoSrho*chi3, -rhoSrho*chi2, -rhoSrho*(chi2*q[IW] - chi3*q[IV]), 0.0, -rhoLn*chi3, rhoLn*chi2, 0.0}; // r\lambda_{+a}
  State rA_m {0.0, 0.0, -rhoSrho*chi3, rhoSrho*chi2,  rhoSrho*(chi2*q[IW] - chi3*q[IV]), 0.0, -rhoLn*chi3, rhoLn*chi2, 0.0}; // r\lambda_{-a}
  // Fast Magnetoacoustic Waves : \lambda_{\pm f}
  auto sigma = [&](const real_t omega) {
    return (omega >= 0.0 ? 1.0 : -1.0);
  };
  const real_t alpha_f = Kokkos::sqrt((a2 - cs2) / (cf2 - cs2));
  const real_t alpha_s = Kokkos::sqrt((cf2 - a2) / (cf2 - cs2));
  const real_t a_ln2 = params.gamma0 / (2.0 * betaLn);
  // const real_t a_ln2 = params.gamma0*(0.5*rhoLn/betaLn)/rhoLn;
  const real_t a_beta = Kokkos::sqrt(0.5 * params.gamma0 / betaAvg);
  const real_t psi_sp = 0.5*alpha_s*rhoLn*u2bar - a_beta*alpha_f*rhoLn*bTrans + (alpha_s*rhoLn*a_ln2)/(params.gamma0 - 1.0) + alpha_s*cs*rhoLn*q[IU] + alpha_f*cf*rhoLn*sigma(b1)*(q[IV]*chi2 + q[IW]*chi3); // \psi_{+s}
  const real_t psi_sm = 0.5*alpha_s*rhoLn*u2bar - a_beta*alpha_f*rhoLn*bTrans + (alpha_s*rhoLn*a_ln2)/(params.gamma0 - 1.0) - alpha_s*cs*rhoLn*q[IU] - alpha_f*cf*rhoLn*sigma(b1)*(q[IV]*chi2 + q[IW]*chi3); // \psi_{-s}
  const real_t psi_fp = 0.5*alpha_f*rhoLn*u2bar + a_beta*alpha_s*rhoLn*bTrans + (alpha_f*rhoLn*a_ln2)/(params.gamma0 - 1.0) + alpha_f*cf*rhoLn*q[IU] - alpha_s*cs*rhoLn*sigma(b1)*(q[IV]*chi2 + q[IW]*chi3); // \psi_{+f}
  const real_t psi_fm = 0.5*alpha_f*rhoLn*u2bar + a_beta*alpha_s*rhoLn*bTrans + (alpha_f*rhoLn*a_ln2)/(params.gamma0 - 1.0) - alpha_f*cf*rhoLn*q[IU] + alpha_s*cs*rhoLn*sigma(b1)*(q[IV]*chi2 + q[IW]*chi3); // \psi_{-f}
  State rF_p {alpha_f*rhoLn, alpha_f*rhoLn*(q[IU] + cf), rhoLn*(alpha_f*q[IV] - alpha_s*cs*chi2*sigma(b1)), rhoLn*(alpha_f*q[IW] - alpha_s*cs*chi3*sigma(b1)), psi_fp, 0.0, alpha_s*a_beta*chi2*Kokkos::sqrt(rhoLn), alpha_s*a_beta*chi3*Kokkos::sqrt(rhoLn), 0.0};
  State rF_m {alpha_f*rhoLn, alpha_f*rhoLn*(q[IU] - cf), rhoLn*(alpha_f*q[IV] + alpha_s*cs*chi2*sigma(b1)), rhoLn*(alpha_f*q[IW] + alpha_s*cs*chi3*sigma(b1)), psi_fm, 0.0, alpha_s*a_beta*chi2*Kokkos::sqrt(rhoLn), alpha_s*a_beta*chi3*Kokkos::sqrt(rhoLn), 0.0};
  // Slow Magnetoacoustic Waves : \lambda_{\pm s}
  State rS_p {alpha_s*rhoLn, alpha_s*rhoLn*(q[IU] + cs), rhoLn*(alpha_s*q[IV] + alpha_f*cf*chi2*sigma(b1)), rhoLn*(alpha_s*q[IW] + alpha_f*cf*chi3*sigma(b1)), psi_sp, 0.0, -alpha_f*a_beta*chi2*Kokkos::sqrt(rhoLn), -alpha_f*a_beta*chi3*Kokkos::sqrt(rhoLn), 0.0};
  State rS_m {alpha_s*rhoLn, alpha_s*rhoLn*(q[IU] - cs), rhoLn*(alpha_s*q[IV] - alpha_f*cf*chi2*sigma(b1)), rhoLn*(alpha_s*q[IW] - alpha_f*cf*chi3*sigma(b1)), psi_sm, 0.0, -alpha_f*a_beta*chi2*Kokkos::sqrt(rhoLn), -alpha_f*a_beta*chi3*Kokkos::sqrt(rhoLn), 0.0};

  // KEPES Flux - eq. (4.72), disspation term
  Matrix R("R", Nfields, Nfields);
  Matrix RT("RTranspose", Nfields, Nfields);
  Matrix LZ("LambdaxZ", Nfields, Nfields);

  for (int i = 0; i < Nfields; ++i) {
    R(i, 0) = rF_p[i]; //- lambda_hat[0]*Z[0];
    R(i, 1) = rA_p[i]; //- lambda_hat[1]*Z[1];
    R(i, 2) = rS_p[i]; //- lambda_hat[2]*Z[2];
    R(i, 3) = rpsi_p[i]; //- lambda_hat[3]*Z[3];
    R(i, 4) = rE[i]; //- lambda_hat[4]*Z[4];
    R(i, 5) = rpsi_m[i]; //- lambda_hat[5]*Z[5];
    R(i, 6) = rS_m[i]; //- lambda_hat[6]*Z[6];
    R(i, 7) = rA_m[i]; //- lambda_hat[7]*Z[7];
    R(i, 8) = rF_m[i]; //- lambda_hat[8]*Z[8];
    // Fill the transpose matrix
    // RT(0, i) = rF_p[i];
    // RT(1, i) = rA_p[i];
    // RT(2, i) = rS_p[i];
    // RT(3, i) = rpsi_p[i];
    // RT(4, i) = rE[i];
    // RT(5, i) = rpsi_m[i];
    // RT(6, i) = rS_m[i];
    // RT(7, i) = rA_m[i];
    // RT(8, i) = rF_m[i];
    for (int j = 0; j < Nfields; ++j) {
      if (i == j) {
        LZ(i, j) = lambda_hat[j] * Z[i]; // Diagonal matrix
      } else {
        LZ(i, j) = 0.0; // Off-diagonal elements are zero
      }
    }
  }
  // auto computeFinalVector = [&] (const Matrix& R, const State& lambda, const State& v, const State& z) {
  //   State finalVector {};

  //   for (int i = 0; i < Nfields; ++i) {
  //       real_t sum = 0.0;
  //       for (int j = 0; j < Nfields; ++j) {
  //           for (int k = 0; k < Nfields; ++k) {
  //               sum += v[j] * R(j, k) * R(i, k) * lambda[k] * z[k];
  //           }
  //       }
  //       finalVector[i] = sum;
  //   }

  //   return finalVector;
  // };
  // D = matmul(R, LZ); // D = R |LAMBDA| Z R^T
  // tmp = matmul(D, RT); // tmp = R^T D
  for (int i = 0; i < Nfields; ++i){
    for (int j = 0; j < Nfields; ++j) {
      RT(j,i) = R(i,j);
      }
  }
  Matrix tmp("tmp", Nfields, Nfields);
  tmp = matmul(R, LZ);
  Matrix tmp2("tmp2", Nfields, Nfields);
  tmp2 = matmul(tmp, RT); // tmp2 = D * R^T
  const State VState = getEntropyJumpStateFromConsStates(qL, qR, params);
  State dissipativeTerm = matvecmul(tmp2, VState); // D * q
  return 0.5*dissipativeTerm;
}

KOKKOS_INLINE_FUNCTION
State getScalarDissipation(State &qL, State &qR, real_t &lambda_max, const DeviceParams &params) {
  State Lmax = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  Lmax *= Kokkos::abs(lambda_max);
  State q = getConsJumpState(qL, qR, params);
  return 0.5 * Lmax * q;
}

KOKKOS_INLINE_FUNCTION
void IdealGLM(State &qL, State &qR, State &flux, real_t &pout, real_t lambda_max, real_t ch, const DeviceParams &params){
  // Ideal GLM MHD Riemann Solver from Derigs et al. 2018  - 10.1016/j.jcp.2018.03.002
  // Compute KEPEC Flux
  FluxKEPEC(qL, qR, flux, pout, ch, params);
  // State dissipative_term = getScalarDissipation(qL, qR, lambda_max, params);
  State dissipative_term = getMatrixDissipation(qL, qR, ch, params);
  flux -= dissipative_term; // Subtract the dissipation term
}

#endif
}

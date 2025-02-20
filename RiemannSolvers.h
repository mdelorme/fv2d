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
KOKKOS_INLINE_FUNCTION
void hlld(State &qL, State &qR, State &flux, real_t &p_gas_out, const Params &params) {
  auto norm2 = [&] (const real_t x, const real_t y, const real_t z){
    return x*x + y*y * z*z;
  };

  auto dot = [&] (const real_t x1, const real_t y1, const real_t z1, const real_t x2, const real_t y2, const real_t z2){
    return x1*x2 + y1*y2 * z1*z2;
  };

  auto FastMagnetocousticSpeed = [&](State &q, const Params &params) {
    const real_t B2 = q[IBX]*q[IBX] + q[IBY]*q[IBY] + q[IBZ]*q[IBZ];
    const real_t lhs = params.gamma * q[IP] + B2;
    const real_t rhs = lhs*lhs - 4.0 * params.gamma * q[IP] * q[IBX]*q[IBX];
    const real_t frac = (lhs + std::sqrt(rhs)) / (2.0 * q[IR]);
    return std::sqrt(frac);
  };

  const real_t rL = qL[IR];
  const real_t uL = qL[IU];
  const real_t vL = qL[IV];
  const real_t wL = qL[IW];
  const real_t pL = qL[IP];
  const real_t BxL = qL[IBX];
  const real_t ByL = qL[IBY];
  const real_t BzL = qL[IBZ];
  const real_t EL = qL[IP]/(params.gamma0-1.0) + 0.5*rL*norm2(uL, vL, wL) + 0.5*norm2(BxL, ByL, BzL);
  const real_t pTotL = qL[IP] + 0.5 * norm2(BxL, ByL, BzL);

  const real_t rR = qR[IR];
  const real_t uR = qR[IU];
  const real_t vR = qR[IV];
  const real_t wR = qR[IW];
  const real_t pR = qR[IP];
  const real_t BxR = qR[IBX]; 
  const real_t ByR = qR[IBY];
  const real_t BzR = qR[IBZ];
  const real_t ER = qR[IP]/(params.gamma0-1.0) + 0.5*rR*norm2(uR, vR, wR) + 0.5*norm2(BxR, ByR, BzR);
  const real_t pTotR = qR[IP] + 0.5 * norm2(BxR, ByR, BzR);

  // ----- Compute waves --------
  const real_t cfL = FastMagnetocousticSpeed(qL, params);
  const real_t cfR = FastMagnetocousticSpeed(qR, params);
  const real_t SL = std::min(uL, uR) - std::max(cfL, cfR);
  const real_t SR = std::max(uL, uR) + std::max(cfL, cfR);
  const real_t SM = ((SR-uR)*rR*uR - (SL-uL)*rL*uL - ptotR + ptotL)/((SR-uR)*rR - (SL-uL)*rL);
  // ----------------------------

  // -----  Common values in * zones --------
  const real_t rLs = rL * (SL-uL)/(SL-SM);
  const real_t rRs = rR * (SR-uR)/(SR-SM);
  const real_t pTots = ((SR-uR)*rR*pTotL - (SL-uL)*rL*pTotR + rL*rR*(SR-uR)*(SL-uL)*(uR-uL))/((SR-uR)*rR - (SL-uL)*rL);

  // ----- Alfvén Waves -----
  const real_t SLs = SM - std::abs(BxL)/std::sqrt(rLs);
  const real_t SRs = SM + std::abs(BxR)/std::sqrt(rRs);

  // ------ Values in *L zone ---------
  const real_t dnmtLs = rL * (SL - uL) * (SL - SM) - BxL*BxL;
  const real_t numMagLs = rL * (SL - uL)*(SL - uL) - BxL*BxL;
  const real_t vLs = vL - BxL*ByL * (SM - uL)/dnmtLs; 
  const real_t wLs = wL - BxL*BzL * (SM - uL)/dnmtLs; 
  const real_t ByLs = ByL * numMagLs/dnmtLs;
  const real_t BzLs = BzL * numMagLs/dnmtLs;
  const real_t ELs = ((SL-uL)*EL - pTotL*uL + pTots*SM + BxL*(dot(uL, vL, wL, BxL, ByL, BzL) - dot(SM, vLs, wLs, BxL, ByLs, BzLs)))/(SL - SM);

  // ------ Values in *R zone ---------
  const real_t dnmtRs = rR * (SR - uR) * (SR - SM) - BxR*BxR;
  const real_t numMagRs = rR * (SR - uR)*(SR - uR) - BxR*BxR;
  const real_t vRs = vR - BxR*ByR * (SM - uR)/dnmtRs; 
  const real_t wRs = wR - BxR*BzR * (SM - uR)/dnmtRs; 
  const real_t ByRs = ByR * numMagRs/dnmtRs;
  const real_t BzRs = BzR * numMagRs/dnmtRs;
  const real_t ERs = ((SR-uR)*ER - pTotR*uR + pTots*SM + BxR*(dot(uR, vR, wR, BxR, ByR, BzR) - dot(SM, vRs, wRs, BxR, ByRs, BzRs)))/(SR - SM);

  // ----- Common values in ** zones -----
  const real_t dnmtss = std::sqrt(rL) + std::sqrt(rR);
  const real_t signBx = std::copysign(BxL, 1);
  const real_t vss = (std::sqrt(rLs)*vLs + std::sqrt(rRs)*vRs + (ByRs-ByLs) * signBx)/dnmtss;
  const real_t wss = (std::sqrt(rLs)*wLs + std::sqrt(rRs)*wRs + (BzRs-BzLs) * signBx)/dnmtss;
  const real_t Byss = (std::sqrt(rLs)*ByRs + std::sqrt(rRs)*ByLs + std::sqrt(rLs*rRs)*(vRs-vLs)*signBx)/(dnmtss);
  const real_t Bzss = (std::sqrt(rLs)*BzRs + std::sqrt(rRs)*BzLs + std::sqrt(rLs*rRs)*(wRs-wLs)*signBx)/(dnmtss);

  const bool is_zerodiv_L = (SM == uL) && (SL==(uL+cfL) || SL==(uL-cfL)) && (ByL==0) && (BzL==0) && (BxL*BxL >= (params.gamma0*pL));
  const bool is_zerodiv_R = (SM == uR) && (SR==(uR+cfR) || SR==(uR-cfR)) && (ByR==0) && (BzR==0) && (BxR*BxR >= (params.gamma0*pR));

  if (is_zerodiv_L){
    vLs = vL;
    wLs = wL;
    ByLs = 0;
    BzLs = 0;
  };

  if (is_zerodiv_R){
    vRs = vR;
    wRs = wR;
    ByRs = 0;
    BzRs = 0;
  };

  // -------- Choice of the flux to compute depending on the zone ------
 
  State Q;
  real_t E; // total energy -- needed to compute the flux

  if (SL > 0) {
     // UL Zone, compute FL 
      Kokkos::deep_copy(Q, qL);
      E = EL;
  } else if (SL <= 0 && 0 <= SL_s) {
    // UL* Zone, compute FL*
      Q[IR] = rLs;
      Q[IU] = SM;
      Q[IV] = vLs;
      Q[IW] = wLs;
      Q[IP] = pTotL - 0.5 * norm2(BxL, ByLs, BzLs);
      Q[IBX] = BxL;
      Q[IBY] = ByL;
      Q[IBZ] = BzL;
      E = ELs;
  } else if (SL_s <= 0 && 0 <= SM) {
    // UL** Zone, compute FL**
      Q[IR] = rLs;
      Q[IU] = SM;
      Q[IV] = vss;
      Q[IW] = wss;
      Q[IP] = pTotL - 0.5 * norm2(BxL, Byss, Bzss);
      Q[IBX] = BxL;
      Q[IBY] = Byss;
      Q[IBZ] = Bzss;
      E = ELs - std::sqrtl(rLs) * (dot(SM, vLs, wLs, BxL, ByLs, BzLs) - dot(SM, vss, wss, BxL, Byss, Bzss)) * std::copysign(1, BxL);
  } else if (SM <= 0 && 0 <= SR_s) {
    // UR** Zone, compute FR**
      Q[IR] = rRs;
      Q[IU] = SM;
      Q[IV] = vss;
      Q[IW] = wss;
      Q[IP] = pTotR - 0.5 * norm2(BxR, Byss, Bzss);
      Q[IBX] = BxR;
      Q[IBY] = Byss;
      Q[IBZ] = Bzss;
      E = ERs + std::sqrtl(rRs) * (dot(SM, vRs, wRs, BxR, ByRs, BzRs) - dot(SM, vss, wss, BxR, Byss, Bzss)) * std::copysign(1, BxR);
  } else if (SR_s <= 0 && 0 <= SR) {
    // UR* Zone, compute FR*
      Q[IR] = rRs;
      Q[IU] = SM;
      Q[IV] = vRs;
      Q[IW] = wRs;
      Q[IP] = pTotR - 0.5 * norm2(BxL, ByRs, BzRs);
      Q[IBX] = BxR;
      Q[IBY] = ByR;
      Q[IBZ] = BzR;
      E = ERs;
  } else if (SR < 0) {
    // UR Zone, compute FR
      Kokkos::deep_copy(Q, qR);
      E = ER; 
  } else {
      throw std::runtime_error("Aucun cas n'a été traité, valeurs d'ondes inattendues");
  }

  flux = computeFlux(Q, E, params);

}

}
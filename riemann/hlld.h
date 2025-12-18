#pragma once

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void hlld(State &qL, State &qR, State &flux, real_t &p_gas_out, const real_t Bx, const DeviceParams &params)
{
  const real_t Bsgn   = (Bx < 0.0 ? -1.0 : 1.0);
  const real_t smalle = std::pow(10.0, -5.0);
  // Deref of states
  const real_t uL  = qL[IU];
  const real_t vL  = qL[IV];
  const real_t wL  = qL[IW];
  const real_t pL  = qL[IP];
  const real_t rL  = qL[IR];
  const real_t ByL = qL[IBY];
  const real_t BzL = qL[IBZ];
  const real_t B2L = Bx * Bx + ByL * ByL + BzL * BzL;
  const real_t pTL = pL + 0.5 * B2L;
  const real_t EL  = pL / (params.gamma0 - 1.0) + 0.5 * rL * (uL * uL + vL * vL + wL * wL) + 0.5 * B2L;

  const real_t uR  = qR[IU];
  const real_t vR  = qR[IV];
  const real_t wR  = qR[IW];
  const real_t pR  = qR[IP];
  const real_t rR  = qR[IR];
  const real_t ByR = qR[IBY];
  const real_t BzR = qR[IBZ];
  const real_t B2R = Bx * Bx + ByR * ByR + BzR * BzR;
  const real_t pTR = pR + 0.5 * B2R;
  const real_t ER  = pR / (params.gamma0 - 1.0) + 0.5 * rR * (uR * uR + vR * vR + wR * wR) + 0.5 * B2R;

  auto computeFastMagnetoAcousticSpeed = [&](const State &q, const DeviceParams &params)
  {
    const real_t gp = params.gamma0 * q[IP];
    const real_t B2 = Bx * Bx + q[IBY] * q[IBY] + q[IBZ] * q[IBZ];

    return sqrt(0.5 * (gp + B2 + sqrt((gp + B2) * (gp + B2) - 4.0 * gp * Bx * Bx)) / q[IR]);
  };

  const real_t cfL = computeFastMagnetoAcousticSpeed(qL, params);
  const real_t cfR = computeFastMagnetoAcousticSpeed(qR, params);

  // HLL Wave speed
  const real_t SL = fmin(uL, uR) - fmax(cfL, cfR);
  const real_t SR = fmax(uL, uR) + fmax(cfL, cfR);

  // Lagrangian speed of sound
  const real_t rCL = rL * (uL - SL);
  const real_t rCR = rR * (SR - uR);

  // Entropy wave speed
  const real_t uS = (rCR * uR + rCL * uL - pTR + pTL) / (rCR + rCL);

  // Single Star state
  const real_t pTS = (rCR * pTL + rCL * pTR - rCR * rCL * (uR - uL)) / (rCR + rCL);

  // Single star densities
  const real_t rLS = rL * (SL - uL) / (SL - uS);
  const real_t rRS = rR * (SR - uR) / (SR - uS);

  // Single star velocities
  const real_t econvL = rL * (SL - uL) * (SL - uS) - Bx * Bx;
  const real_t econvR = rR * (SR - uR) * (SR - uS) - Bx * Bx;

  const real_t uconvL = (uS - uL) / econvL;
  const real_t uconvR = (uS - uR) / econvR;
  const real_t BconvL = (rCL * rCL / rL - Bx * Bx) / econvL;
  const real_t BconvR = (rCR * rCR / rR - Bx * Bx) / econvR;

  real_t vLS, vRS, wLS, wRS, ByLS, ByRS, BzLS, BzRS;

  // Switching to two state on the left ?
  if (std::abs(econvL) < smalle * Bx * Bx)
  {
    vLS  = vL;
    wLS  = wL;
    ByLS = ByL;
    BzLS = BzL;
  }
  else
  {
    vLS  = vL - Bx * ByL * uconvL;
    wLS  = wL - Bx * BzL * uconvL;
    ByLS = ByL * BconvL;
    BzLS = BzL * BconvL;
  }

  // Switching to two state on the right ?
  if (std::abs(econvR) < smalle * Bx * Bx)
  {
    vRS  = vR;
    wRS  = wR;
    ByRS = ByR;
    BzRS = BzR;
  }
  else
  {
    vRS  = vR - Bx * ByR * uconvR;
    wRS  = wR - Bx * BzR * uconvR;
    ByRS = ByR * BconvR;
    BzRS = BzR * BconvR;
  }

  // Single star total energy
  const real_t udotBL   = uL * Bx + vL * ByL + wL * BzL;
  const real_t udotBR   = uR * Bx + vR * ByR + wR * BzR;
  const real_t uSdotBSL = uS * Bx + vLS * ByLS + wLS * BzLS;
  const real_t uSdotBSR = uS * Bx + vRS * ByRS + wRS * BzRS;

  const real_t ELS = ((SL - uL) * EL - pTL * uL + pTS * uS + Bx * (udotBL - uSdotBSL)) / (SL - uS);
  const real_t ERS = ((SR - uR) * ER - pTR * uR + pTS * uS + Bx * (udotBR - uSdotBSR)) / (SR - uS);

  // Alfven wave speeds
  const real_t srLS = sqrt(rLS);
  const real_t srRS = sqrt(rRS);
  const real_t SLS  = uS - fabs(Bx) / srLS;
  const real_t SRS  = uS + fabs(Bx) / srRS;

  // Double Star state
  const real_t den_fac = 1.0 / (srLS + srRS);
  const real_t vSS     = (srLS * vLS + srRS * vRS + (ByRS - ByLS) * Bsgn) * den_fac;
  const real_t wSS     = (srLS * wLS + srRS * wRS + (BzRS - BzLS) * Bsgn) * den_fac;
  const real_t BySS    = (srLS * ByRS + srRS * ByLS + srLS * srRS * (vRS - vLS) * Bsgn) * den_fac;
  const real_t BzSS    = (srLS * BzRS + srRS * BzLS + srLS * srRS * (wRS - wLS) * Bsgn) * den_fac;

  const real_t uSSdotBSS = uS * Bx + vSS * BySS + wSS * BzSS;

  const real_t ELSS = ELS - srLS * (uSdotBSL - uSSdotBSS) * Bsgn;
  const real_t ERSS = ERS + srRS * (uSdotBSR - uSSdotBSS) * Bsgn;

  // Lambda to compute a flux from a primitive state
  auto computeFlux = [&](const State &q, const real_t e_tot) -> State
  {
    State res{};

    res[IR]   = q[IR] * q[IU];
    res[IU]   = q[IR] * q[IU] * q[IU] + q[IP] - q[IBX] * q[IBX];
    res[IV]   = q[IR] * q[IU] * q[IV] - q[IBX] * q[IBY];
    res[IW]   = q[IR] * q[IU] * q[IW] - q[IBX] * q[IBZ];
    res[IBX]  = 0.0;
    res[IBY]  = q[IBY] * q[IU] - q[IBX] * q[IV];
    res[IBZ]  = q[IBZ] * q[IU] - q[IBX] * q[IW];
    res[IE]   = (e_tot + q[IP]) * q[IU] - q[IBX] * (q[IBX] * q[IU] + q[IBY] * q[IV] + q[IBZ] * q[IW]);
    res[IPSI] = 0.0;
    return res;
  };

  // Disjunction of cases

  State q;
  real_t e_tot;
  if (SL > 0.0)
  { // qL
    q     = qL;
    e_tot = EL;
    q[IP] = pTL;

    p_gas_out = qL[IP];
  }
  else if (SLS > 0.0)
  { // qL*
    q[IR]  = rLS;
    q[IU]  = uS;
    q[IV]  = vLS;
    q[IW]  = wLS;
    q[IBX] = Bx;
    q[IBY] = ByLS;
    q[IBZ] = BzLS;

    q[IP] = pTS;
    e_tot = ELS;

    p_gas_out = qL[IP];
  }
  else if (uS > 0.0)
  { // qL**
    q[IR]  = rLS;
    q[IU]  = uS;
    q[IV]  = vSS;
    q[IW]  = wSS;
    q[IBX] = Bx;
    q[IBY] = BySS;
    q[IBZ] = BzSS;

    q[IP] = pTS;
    e_tot = ELSS;

    p_gas_out = qL[IP];
  }
  else if (SRS > 0.0)
  { // qR**
    q[IR]  = rRS;
    q[IU]  = uS;
    q[IV]  = vSS;
    q[IW]  = wSS;
    q[IBX] = Bx;
    q[IBY] = BySS;
    q[IBZ] = BzSS;

    q[IP] = pTS;
    e_tot = ERSS;

    p_gas_out = qR[IP];
  }
  else if (SR > 0.0)
  { // qR*
    q[IR]  = rRS;
    q[IU]  = uS;
    q[IV]  = vRS;
    q[IW]  = wRS;
    q[IBX] = Bx;
    q[IBY] = ByRS;
    q[IBZ] = BzRS;

    q[IP] = pTS;
    e_tot = ERS;

    p_gas_out = qR[IP];
  }
  else
  { // SR < 0.0; qR
    q     = qR;
    e_tot = ER;
    q[IP] = pTR;

    p_gas_out = qR[IP];
  }

  flux = computeFlux(q, e_tot);
}
} // namespace fv2d

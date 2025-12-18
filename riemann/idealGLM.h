#pragma once

#include "../LinAlg.h"

namespace fv2d
{
KOKKOS_INLINE_FUNCTION
void FluxDerigs(const State &qL, const State &qR, State &flux, real_t ch, const DeviceParams &params)
{
  const State qAvg   = 0.5 * (qL + qR);
  const real_t rhoLn = logMean(qL[IR], qR[IR]);

  const real_t betaL = 0.5 * qL[IR] / qL[IP];
  const real_t betaR = 0.5 * qR[IR] / qR[IP];
  // const real_t betaAvg = 0.5 * qAvg[IR]/qAvg[IP];
  // const real_t betaLn = 0.5 * rhoLn / logMean(qL[IP], qR[IP]);
  const real_t betaAvg = 0.5 * (betaL + betaR);
  const real_t betaLn  = logMean(betaL, betaR);

  // sum_i {{Bi^2}}
  const real_t B2Avg = 0.5 * (qL[IBX] * qL[IBX] + qR[IBX] * qR[IBX] + qL[IBY] * qL[IBY] + qR[IBY] * qR[IBY] +
                              qL[IBZ] * qL[IBZ] + qR[IBZ] * qR[IBZ]);
  // sum_i {{vi^2}}
  const real_t v2Avg =
      0.5 * (qL[IU] * qL[IU] + qR[IU] * qR[IU] + qL[IV] * qL[IV] + qR[IV] * qR[IV] + qL[IW] * qL[IW] + qR[IW] * qR[IW]);
  // sum_i {{vi*Bi^2}}
  const real_t uB2Avg = 0.5 * (qL[IU] * qL[IBX] * qL[IBX] + qR[IU] * qR[IBX] * qR[IBX] + qL[IU] * qL[IBY] * qL[IBY] +
                               qR[IU] * qR[IBY] * qR[IBY] + qL[IU] * qL[IBZ] * qL[IBZ] + qR[IU] * qR[IBZ] * qR[IBZ]);
  // sum_i {{vi*Bi}}
  const real_t vBAvg = 0.5 * (qL[IU] * qL[IBX] + qR[IU] * qR[IBX] + qL[IV] * qL[IBY] + qR[IV] * qR[IBY] +
                              qL[IW] * qL[IBZ] + qR[IW] * qR[IBZ]);

  const real_t ptot = 0.5 * qAvg[IR] / betaAvg + 0.5 * B2Avg;

  // Flux components
  flux[IR]   = rhoLn * qAvg[IU];
  flux[IU]   = flux[IR] * qAvg[IU] - qAvg[IBX] * qAvg[IBX] + ptot;
  flux[IV]   = flux[IR] * qAvg[IV] - qAvg[IBX] * qAvg[IBY];
  flux[IW]   = flux[IR] * qAvg[IW] - qAvg[IBX] * qAvg[IBZ];
  flux[IBX]  = ch * qAvg[IPSI];
  flux[IBY]  = qAvg[IU] * qAvg[IBY] - qAvg[IV] * qAvg[IBX]; // Dedner
  flux[IBZ]  = qAvg[IU] * qAvg[IBZ] - qAvg[IW] * qAvg[IBX]; // Dedner
  flux[IPSI] = ch * qAvg[IBX];

  flux[IE] = flux[IR] * (1.0 / (2.0 * betaLn * (params.gamma0 - 1.0)) - 0.5 * v2Avg) + flux[IU] * qAvg[IU] +
             flux[IV] * qAvg[IV] + flux[IW] * qAvg[IW] + flux[IBX] * qAvg[IBX] + flux[IBY] * qAvg[IBY] +
             flux[IBZ] * qAvg[IBZ] + flux[IPSI] * qAvg[IPSI] + qAvg[IBX] * vBAvg - 0.5 * uB2Avg -
             ch * 0.5 * (qL[IBX] * qL[IPSI] + qR[IBX] * qR[IPSI]);
}

KOKKOS_INLINE_FUNCTION
void FluxChandrashekar(const State &qL, const State &qR, State &flux, real_t ch, const DeviceParams &params)
{
  const State qAvg   = 0.5 * (qL + qR);
  const real_t rhoLn = logMean(qL[IR], qR[IR]);

  const real_t betaL   = 0.5 * qL[IR] / qL[IP];
  const real_t betaR   = 0.5 * qR[IR] / qR[IP];
  const real_t betaAvg = 0.5 * qAvg[IR] / qAvg[IP];
  const real_t betaLn  = 0.5 * rhoLn / logMean(qL[IP], qR[IP]);
  // sum_i {{Bi^2}}
  const real_t B2Avg = 0.5 * (qL[IBX] * qL[IBX] + qR[IBX] * qR[IBX] + qL[IBY] * qL[IBY] + qR[IBY] * qR[IBY] +
                              qL[IBZ] * qL[IBZ] + qR[IBZ] * qR[IBZ]);
  // sum_i {{vi^2}}
  const real_t v2Avg =
      0.5 * (qL[IU] * qL[IU] + qR[IU] * qR[IU] + qL[IV] * qL[IV] + qR[IV] * qR[IV] + qL[IW] * qL[IW] + qR[IW] * qR[IW]);

  const real_t vBAvg = qAvg[IU] * qAvg[IBX] + qAvg[IV] * qAvg[IBY] + qAvg[IW] * qAvg[IBZ];
  // Average of the magnetic flux
  const real_t ptot       = 0.5 * qAvg[IR] / betaAvg + 0.5 * B2Avg;
  const real_t betaUAvg   = 0.5 * (qL[IU] * betaL + qR[IU] * betaR);
  const real_t betaVAvg   = 0.5 * (qL[IV] * betaL + qR[IV] * betaR);
  const real_t betaWAvg   = 0.5 * (qL[IW] * betaL + qR[IW] * betaR);
  const real_t betaPsiAvg = 0.5 * (qL[IPSI] * betaL + qR[IPSI] * betaR);

  // Flux components
  flux[IR]   = rhoLn * qAvg[IU];
  flux[IU]   = flux[IR] * qAvg[IU] - qAvg[IBX] * qAvg[IBX] + ptot;
  flux[IV]   = flux[IR] * qAvg[IV] - qAvg[IBX] * qAvg[IBY];
  flux[IW]   = flux[IR] * qAvg[IW] - qAvg[IBX] * qAvg[IBZ];
  flux[IBX]  = ch * betaPsiAvg / betaAvg; // Barbier
  flux[IBY]  = (1.0 / betaAvg) * (betaUAvg * qAvg[IBY] - betaVAvg * qAvg[IBX]);
  flux[IBZ]  = (1.0 / betaAvg) * (betaUAvg * qAvg[IBZ] - betaWAvg * qAvg[IBX]);
  flux[IPSI] = ch * qAvg[IBX];

  flux[IE] = flux[IR] * (1.0 / (2.0 * betaLn * (params.gamma0 - 1.0)) - 0.5 * v2Avg) + flux[IU] * qAvg[IU] +
             flux[IV] * qAvg[IV] + flux[IW] * qAvg[IW] + flux[IBX] * qAvg[IBX] + flux[IBY] * qAvg[IBY] +
             flux[IBZ] * qAvg[IBZ] + flux[IPSI] * qAvg[IPSI] + qAvg[IBX] * vBAvg - 0.5 * qAvg[IU] * B2Avg -
             ch * 0.5 * (qL[IBX] * qL[IPSI] + qR[IBX] * qR[IPSI]);
}

KOKKOS_INLINE_FUNCTION
void FluxKEPEC(const State &qL, const State &qR, State &flux, real_t &pout, real_t ch, const DeviceParams &params)
{
  // Note : In fv2d, the vector q holds the primitive variables: q = (rho, u, v, w, p, Bx, By, Bz, Psi)
  FluxDerigs(qL, qR, flux, ch, params);
  // FluxChandrashekar(qL, qR, flux, ch, params);
  pout = 0.5 * (qL[IP] + qR[IP]);
}

KOKKOS_INLINE_FUNCTION
State getMatrixDissipation(const State &qL, const State &qR, real_t ch, const DeviceParams &params)
{
  real_t u2_L, u2_R, p_L, p_R, beta_L, beta_R, rho_L, rho_R;
  real_t u_L, v_L, w_L, u_R, v_R, w_R;
  real_t Bx_L, By_L, Bz_L, Bx_R, By_R, Bz_R, psi_L, psi_R;
  rho_L = qL[IR];
  rho_R = qR[IR];
  // real_t srho_R = 1./rho_R;
  // real_t srho_L = 1./rho_L;
  u_L   = qL[IU];
  v_L   = qL[IV];
  w_L   = qL[IW];
  u_R   = qR[IU];
  v_R   = qR[IV];
  w_R   = qR[IW];
  Bx_L  = qL[IBX];
  By_L  = qL[IBY];
  Bz_L  = qL[IBZ];
  Bx_R  = qR[IBX];
  By_R  = qR[IBY];
  Bz_R  = qR[IBZ];
  psi_L = qL[IPSI];
  psi_R = qR[IPSI];
  // u2_L  = u_L * u_L + v_L * v_L + w_L * w_L;
  // u2_R  = u_R * u_R + v_R * v_R + w_R * w_R;
  // real_t B2_R = Bx_R*Bx_R + By_R*By_R + Bz_R*Bz_R;
  // real_t B2_L = Bx_L*Bx_L + By_L*By_L + Bz_L*Bz_L;
  p_L    = qL[IP];
  p_R    = qR[IP];
  beta_L = 0.5 * rho_L / p_L; //! beta=rho/(2*p)
  beta_R = 0.5 * rho_R / p_R;

  // ! Get the averages for the numerical flux
  real_t rhoLN   = logMean(rho_L, rho_R);
  real_t sbetaLN = 1. / logMean(beta_L, beta_R);
  // uAvg       = 0.5 * (u_L +  u_R);
  // real_t u2Avg = 0.5 * (u2_L + u2_R);
  // BAvg       = 0.5 * (B_L +  B_R);
  // real_t B2Avg      = 0.5 * (B2_L + B2_R);
  // real_t u1_B2Avg   = 0.5 * (u_L*B2_L + u_R*B2_R);
  // real_t uB_Avg     = 0.5 * (u_L*Bx_L + v_L*By_L + w_L*Bz_L  + u_R*Bx_R + v_R*By_R + w_R*Bz_R);
  real_t betaAvg = 0.5 * (beta_L + beta_R);
  real_t pAvg    = 0.5 * (rho_L + rho_R) / (beta_L + beta_R); //! rhoMEAN/(2*betaMEAN)
  real_t psiAvg  = 0.5 * (psi_L + psi_R);

  // real_t pTilde     = pAvg + 0.5 * B2Avg; //!+1/(2mu_0)({{|B|^2}}...)
  Vect uAvg = {0.5 * (u_L + u_R), 0.5 * (v_L + v_R), 0.5 * (w_L + w_R)};
  Vect Bavg = {0.5 * (Bx_L + Bx_R), 0.5 * (By_L + By_R), 0.5 * (Bz_L + Bz_R)};
  // ! Entropy conserving and kinetic energy conserving flux
  // flux[IR] = rhoLN*uAvg[IX]
  // flux[IU] = flux[IR]*uAvg[IX]- 0.5*Bavg[IX]*Bavg[IX] + pTilde;
  // flux[IV] = flux[IR]*uAvg[IY] - 0.5*Bavg[IX]*Bavg[IY];
  // flux[IW] = flux[IR]*uAvg[IZ] - 0.5*Bavg[IX]*Bavg[IZ];
  // flux[IBY] = uAvg[IX]*Bavg[IY] - Bavg[IX]*uAvg[IY];
  // flux[IBZ] = uAvg[IX]*Bavg[IZ] - Bavg[IX]*uAvg[IZ];
  // flux[IBX] = ch*psiAvg;
  // flux[IPSI] = ch*Bavg[IX];

  // flux[IE] = flux[IR]*0.5*(skappaM1*sbetaLN - u2Avg)
  //            + (uAvg(IX)*flux[IU] + uAvg(IY)*flux[IV] + uAvg(IZ)*flux[IW])
  //            + (Bavg[IX]*flux[IBX] + Bavg[IY]*flux[IBY] + Bavg[IZ]*flux[IBZ])
  //            - 0.5*u1_B2Avg +Bavg[IX]*uB_Avg
  //            + flux[IPSI]*psiAvg-ch*0.5*(B_L(1)*psi_L+B_R(1)*psi_R);
  // ! MHD waves selective dissipation

  // ! Compute additional averages
  real_t rhoAvg = 0.5 * (rho_L + rho_R);
  real_t pLN    = 0.5 * rhoLN * sbetaLN;
  real_t u2Avg  = u_L * u_R + v_L * v_R + w_L * w_R; //! Redefinition of u2Avg -> \bar{\norm{u}^2}
  // real_t srhoLN = 1./rhoLN;
  real_t aA    = Kokkos::sqrt(params.gamma0 * pAvg / rhoLN);
  real_t aLN   = Kokkos::sqrt(params.gamma0 * pLN / rhoLN);
  real_t abeta = Kokkos::sqrt(0.5 * params.gamma0 / betaAvg);
  real_t bb1A  = Bavg[IX] / Kokkos::sqrt(rhoLN);
  real_t bb2A  = Bavg[IY] / Kokkos::sqrt(rhoLN);
  real_t bb3A  = Bavg[IZ] / Kokkos::sqrt(rhoLN);
  real_t bbA   = Kokkos::sqrt(bb1A * bb1A + bb2A * bb2A + bb3A * bb3A);
  real_t aabbA = aA * aA + bbA * bbA;

  // ! Alfven speed
  // ! Derigs et al. (2018), (4.63)
  real_t ca = Kokkos::abs(bb1A);

  // ! Control round-off errors
  real_t xx = aabbA * aabbA - 4.0 * aA * aA * bb1A * bb1A;
  if (xx <= 0.0)
    xx = 0.0;
  real_t xxx = aabbA + Kokkos::sqrt(xx);

  // ! Fast magnetoacoustic speed
  // ! Derigs et al. (2018), (4.63)
  real_t cf = Kokkos::sqrt(0.5 * xxx);

  // ! Control round-off errors
  xxx = aabbA - Kokkos::sqrt(xx);
  if (xxx <= 0.0)
    xxx = 0.0;

  // ! Slow magnetoacoustic speed
  // ! Derigs et al. (2018), (4.63)
  real_t cs = Kokkos::sqrt(0.5 * xxx);

  real_t bperpA = Kokkos::sqrt(bb2A * bb2A + bb3A * bb3A);
  // ! In case of very small bperpA, the values of betaA_{2,3}
  // ! are indeterminable so we make them pairwise orthogonal
  // ! Derigs et al. (2018), (4.63)
  real_t beta2A, beta3A;
  if (bperpA >= 1.0E-14)
  {
    beta2A = bb2A / bperpA;
    beta3A = bb3A / bperpA;
  }
  else
  {
    bperpA = 0.0;
    beta2A = 1.0 / Kokkos::sqrt(2.0);
    beta3A = 1.0 / Kokkos::sqrt(2.0);
  }

  // ! Avoid negative round-off errors when computing alphaf (auxiliary variable)
  xx = 0.0;
  real_t alphaf, alphas;
  if ((cf * cf - cs * cs) >= 0.0)
    xx = (aA * aA - cs * cs) / (cf * cf - cs * cs);
  if (xx >= 0.0)
  {
    alphaf = Kokkos::sqrt(xx);
  }
  else
  {
    alphaf = 0.0;
  }
  // ! Avoid negative round-off errors when computing alphas (auxiliary variable)
  xx = 0.0;
  if ((cf * cf - cs * cs) >= 0.0)
    xx = (cf * cf - aA * aA) / (cf * cf - cs * cs);
  if (xx >= 0.0)
  {
    alphas = Kokkos::sqrt(xx);
  }
  else
  {
    alphas = 0.0;
  }

  const real_t sgnb1 = (bb1A < 0.0) ? -1.0 : 1.0;

  const real_t skappaM1 = 1.0 / (params.gamma0 - 1.0);
  // ! Derigs et al. (2018), (4.63)
  const real_t psiSplus = 0.5 * alphas * rhoLN * u2Avg - abeta * alphaf * rhoLN * bperpA +
                          alphas * rhoLN * aLN * aLN * skappaM1 + alphas * cs * rhoLN * uAvg[IX] +
                          alphaf * cf * rhoLN * sgnb1 * (uAvg[IY] * beta2A + uAvg[IZ] * beta3A);
  const real_t psiSminus = 0.5 * alphas * rhoLN * u2Avg - abeta * alphaf * rhoLN * bperpA +
                           alphas * rhoLN * aLN * aLN * skappaM1 - alphas * cs * rhoLN * uAvg[IX] -
                           alphaf * cf * rhoLN * sgnb1 * (uAvg[IY] * beta2A + uAvg[IZ] * beta3A);

  const real_t psiFplus = 0.5 * alphaf * rhoLN * u2Avg + abeta * alphas * rhoLN * bperpA +
                          alphaf * rhoLN * aLN * aLN * skappaM1 + alphaf * cf * rhoLN * uAvg[IX] -
                          alphas * cs * rhoLN * sgnb1 * (uAvg[IY] * beta2A + uAvg[IZ] * beta3A);
  const real_t psiFminus = 0.5 * alphaf * rhoLN * u2Avg + abeta * alphas * rhoLN * bperpA +
                           alphaf * rhoLN * aLN * aLN * skappaM1 - alphaf * cf * rhoLN * uAvg[IX] +
                           alphas * cs * rhoLN * sgnb1 * (uAvg[IY] * beta2A + uAvg[IZ] * beta3A);

  // ! + fast magnetoacoustic wave
  // ! Derigs et al. (2018), (4.68)
  Matrix Rmatrix{};
  Rmatrix[0][0] = alphaf * rhoLN;
  Rmatrix[1][0] = alphaf * rhoLN * (uAvg[IX] + cf);
  Rmatrix[2][0] = rhoLN * (alphaf * uAvg[IY] - alphas * cs * beta2A * sgnb1);
  Rmatrix[3][0] = rhoLN * (alphaf * uAvg[IZ] - alphas * cs * beta3A * sgnb1);
  Rmatrix[4][0] = psiFplus;
  Rmatrix[5][0] = 0.0;
  Rmatrix[6][0] = alphas * abeta * beta2A * Kokkos::sqrt(rhoLN);
  Rmatrix[7][0] = alphas * abeta * beta3A * Kokkos::sqrt(rhoLN);
  Rmatrix[8][0] = 0.0;

  // ! + Alfven wave
  // ! Derigs et al. (2018), (4.67)
  Rmatrix[0][1] = 0.0;
  Rmatrix[1][1] = 0.0;
  Rmatrix[2][1] = rhoLN * Kokkos::sqrt(rhoAvg) * beta3A;
  Rmatrix[3][1] = -rhoLN * Kokkos::sqrt(rhoAvg) * beta2A;
  Rmatrix[4][1] = -rhoLN * Kokkos::sqrt(rhoAvg) * (beta2A * uAvg[IZ] - beta3A * uAvg[IY]);
  Rmatrix[5][1] = 0.0;
  Rmatrix[6][1] = -rhoLN * beta3A;
  Rmatrix[7][1] = rhoLN * beta2A;
  Rmatrix[8][1] = 0.0;

  // ! + slow magnetoacoustic wave
  // ! Derigs et al. (2018), (4.69)
  Rmatrix[0][2] = alphas * rhoLN;
  Rmatrix[1][2] = alphas * rhoLN * (uAvg[IX] + cs);
  Rmatrix[2][2] = rhoLN * (alphas * uAvg[IY] + alphaf * cf * beta2A * sgnb1);
  Rmatrix[3][2] = rhoLN * (alphas * uAvg[IZ] + alphaf * cf * beta3A * sgnb1);
  Rmatrix[4][2] = psiSplus;
  Rmatrix[5][2] = 0.0;
  Rmatrix[6][2] = -alphaf * abeta * beta2A * Kokkos::sqrt(rhoLN);
  Rmatrix[7][2] = -alphaf * abeta * beta3A * Kokkos::sqrt(rhoLN);
  Rmatrix[8][2] = 0.0;

  // ! + GLM wave
  // ! Dergs et al. (2018), eq. (4.65)
  Rmatrix[0][3] = 0.0;
  Rmatrix[1][3] = 0.0;
  Rmatrix[2][3] = 0.0;
  Rmatrix[3][3] = 0.0;
  Rmatrix[4][3] = Bavg[IX] + psiAvg;
  Rmatrix[5][3] = 1.0;
  Rmatrix[6][3] = 0.0;
  Rmatrix[7][3] = 0.0;
  Rmatrix[8][3] = 1.0;

  // ! Entropy wave
  // ! Derigs et al. (2018), (4.66)
  Rmatrix[0][4] = 1.0;
  Rmatrix[1][4] = uAvg[IX];
  Rmatrix[2][4] = uAvg[IY];
  Rmatrix[3][4] = uAvg[IZ];
  Rmatrix[4][4] = 0.5 * u2Avg;
  Rmatrix[5][4] = 0.0;
  Rmatrix[6][4] = 0.0;
  Rmatrix[7][4] = 0.0;
  Rmatrix[8][4] = 0.0;

  // ! - GLM wave
  // ! Dergs et al. (2018), eq. (4.65)
  Rmatrix[0][5] = 0.0;
  Rmatrix[1][5] = 0.0;
  Rmatrix[2][5] = 0.0;
  Rmatrix[3][5] = 0.0;
  Rmatrix[4][5] = Bavg[IX] - psiAvg;
  Rmatrix[5][5] = 1.0;
  Rmatrix[6][5] = 0.0;
  Rmatrix[7][5] = 0.0;
  Rmatrix[8][5] = -1.0;

  // ! - slow magnetoacoustic wave
  // ! Derigs et al. (2018), (4.69)
  Rmatrix[0][6] = alphas * rhoLN;
  Rmatrix[1][6] = alphas * rhoLN * (uAvg[IX] - cs);
  Rmatrix[2][6] = rhoLN * (alphas * uAvg[IY] - alphaf * cf * beta2A * sgnb1);
  Rmatrix[3][6] = rhoLN * (alphas * uAvg[IZ] - alphaf * cf * beta3A * sgnb1);
  Rmatrix[4][6] = psiSminus;
  Rmatrix[5][6] = 0.0;
  Rmatrix[6][6] = -alphaf * abeta * beta2A * Kokkos::sqrt(rhoLN);
  Rmatrix[7][6] = -alphaf * abeta * beta3A * Kokkos::sqrt(rhoLN);
  Rmatrix[8][6] = 0.0;

  // ! - Alfven wave
  // ! Derigs et al. (2018), (4.67)
  Rmatrix[0][7] = 0.0;
  Rmatrix[1][7] = 0.0;
  Rmatrix[2][7] = -rhoLN * Kokkos::sqrt(rhoAvg) * beta3A;
  Rmatrix[3][7] = rhoLN * Kokkos::sqrt(rhoAvg) * beta2A;
  Rmatrix[4][7] = rhoLN * Kokkos::sqrt(rhoAvg) * (beta2A * uAvg[IZ] - beta3A * uAvg[IY]);
  Rmatrix[5][7] = 0.0;
  Rmatrix[6][7] = -rhoLN * beta3A;
  Rmatrix[7][7] = rhoLN * beta2A;
  Rmatrix[8][7] = 0.0;

  // ! - fast magnetoacoustic wave
  // ! Derigs et al. (2018), (4.68)
  Rmatrix[0][8] = alphaf * rhoLN;
  Rmatrix[1][8] = alphaf * rhoLN * (uAvg[IX] - cf);
  Rmatrix[2][8] = rhoLN * (alphaf * uAvg[IY] + alphas * cs * beta2A * sgnb1);
  Rmatrix[3][8] = rhoLN * (alphaf * uAvg[IZ] + alphas * cs * beta3A * sgnb1);
  Rmatrix[4][8] = psiFminus;
  Rmatrix[5][8] = 0.0;
  Rmatrix[6][8] = alphas * abeta * beta2A * Kokkos::sqrt(rhoLN);
  Rmatrix[7][8] = alphas * abeta * beta3A * Kokkos::sqrt(rhoLN);
  Rmatrix[8][8] = 0.0;

  // ! A blend of the 9waves solver and LLF
  real_t phi = Kokkos::sqrt(Kokkos::abs(1.0 - (p_R * rho_R / (p_L * rho_L))) / (1.0 + (p_R * rho_R / (p_L * rho_L))));
  real_t LambdaMax = Kokkos::max(Kokkos::abs(uAvg[IX] + cf), Kokkos::abs(uAvg[IX] - cf));

  State Dmatrix{};
  Dmatrix[0] = (1. - phi) * Kokkos::abs(uAvg[IX] + cf) + phi * LambdaMax;
  Dmatrix[1] = (1. - phi) * Kokkos::abs(uAvg[IX] + ca) + phi * LambdaMax;
  Dmatrix[2] = (1. - phi) * Kokkos::abs(uAvg[IX] + cs) + phi * LambdaMax;
  Dmatrix[3] = (1. - phi) * Kokkos::abs(uAvg[IX] + ch) + phi * LambdaMax;
  Dmatrix[4] = (1. - phi) * Kokkos::abs(uAvg[IX]) + phi * LambdaMax;
  Dmatrix[5] = (1. - phi) * Kokkos::abs(uAvg[IX] - ch) + phi * LambdaMax;
  Dmatrix[6] = (1. - phi) * Kokkos::abs(uAvg[IX] - cs) + phi * LambdaMax;
  Dmatrix[7] = (1. - phi) * Kokkos::abs(uAvg[IX] - ca) + phi * LambdaMax;
  Dmatrix[8] = (1. - phi) * Kokkos::abs(uAvg[IX] - cf) + phi * LambdaMax;

  //     ! Pure 9waves solver (no LLF)
  // Dmatrix[0] = Kokkos::abs( uAvg[IX] + cf ); // ! + fast magnetoacoustic wave
  // Dmatrix[1] = Kokkos::abs( uAvg[IX] + ca ); //! + Alfven wave
  // Dmatrix[2] = Kokkos::abs( uAvg[IX] + cs ); //! + slow magnetoacoustic wave
  // Dmatrix[3] = Kokkos::abs( uAvg[IX] + ch ); //! + GLM wave
  // Dmatrix[4] = Kokkos::abs( uAvg[IX]      ); //! / Entropy wave
  // Dmatrix[5] = Kokkos::abs( uAvg[IX] - ch ); //! - GLM wave
  // Dmatrix[6] = Kokkos::abs( uAvg[IX] - cs ); //! - slow magnetoacoustic wave
  // Dmatrix[7] = Kokkos::abs( uAvg[IX] - ca ); //! - Alfven wave
  // Dmatrix[8] = Kokkos::abs( uAvg[IX] - cf ); //! - fast magnetoacoustic wave

  // ! Diagonal scaling matrix as described in Winters et al., eq. (4.15)
  // Tmatrix = 0.0
  State Tmatrix{};
  Tmatrix[0] = 0.5 / params.gamma0 / rhoLN;                   //! + f
  Tmatrix[1] = 0.25 / betaAvg / rhoLN / rhoLN;                //! + a
  Tmatrix[2] = Tmatrix[0];                                    //! + s
  Tmatrix[3] = 0.25 / betaAvg;                                //! + GLM
  Tmatrix[4] = rhoLN * (params.gamma0 - 1.0) / params.gamma0; //! E
  Tmatrix[5] = Tmatrix[3];                                    //! - GLM
  Tmatrix[6] = Tmatrix[0];                                    //! - s
  Tmatrix[7] = Tmatrix[1];                                    //! - a
  Tmatrix[8] = Tmatrix[0];                                    //! - f

  // ! Scale D matrix
  Dmatrix      = Dmatrix * Tmatrix;
  Matrix RT    = transpose(Rmatrix);
  State V_jump = getEntropyJumpState(qL, qR, params);
  State res    = matvecmul(Rmatrix, Dmatrix * matvecmul(RT, V_jump));

  return res;
}

KOKKOS_INLINE_FUNCTION
State getScalarDissipation(const State &qL, const State &qR, const DeviceParams &params)
{
  State Lmax{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  State q                     = 0.5 * (qL + qR);
  const real_t lambdaMaxLocal = Kokkos::max(Kokkos::abs(q[IU] - fastMagnetoAcousticSpeed(q, params, IX)),
                                            Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX)));
  Lmax *= lambdaMaxLocal;
  State qJ = getConsJumpState(qL, qR, params);
  return Lmax * qJ;
}

// Ideal GLM MHD Riemann Solver from Derigs et al. 2018  - 10.1016/j.jcp.2018.03.002
KOKKOS_INLINE_FUNCTION
void IdealGLM(const State &qL, const State &qR, State &flux, real_t &pout, real_t ch, const DeviceParams &params)
{
  // 1. Compute KEPEC Flux
  FluxKEPEC(qL, qR, flux, pout, ch, params);
  // 2. Compute the dissipation term for the KEPES flux
  // State dissipative_term = getScalarDissipation(qL, qR, params);
  State dissipative_term = getMatrixDissipation(qL, qR, ch, params);
  flux -= 0.5 * dissipative_term;
}

} // namespace fv2d

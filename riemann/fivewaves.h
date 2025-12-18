#pragma once

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void FiveWaves(State &qL, State &qR, State &flux, real_t &pout, const DeviceParams &params)
{
  const uint IZ            = 2;
  constexpr real_t epsilon = 1.0e-16;
  const real_t B2L         = qL[IBX] * qL[IBX] + qL[IBY] * qL[IBY] + qL[IBZ] * qL[IBZ];
  const real_t B2R         = qR[IBX] * qR[IBX] + qR[IBY] * qR[IBY] + qR[IBZ] * qR[IBZ];
  const Vect pL{-qL[IBX] * qL[IBX] + qL[IP] + B2L / 2, -qL[IBX] * qL[IBY], -qL[IBX] * qL[IBZ]};

  const Vect pR{-qR[IBX] * qR[IBX] + qR[IP] + B2R / 2, -qR[IBX] * qR[IBY], -qR[IBX] * qR[IBZ]};

  // 1. Compute speeds
  const real_t csL = speedOfSound(qL, params);
  const real_t csR = speedOfSound(qR, params);
  real_t caL       = sqrt(qL[IR] * (qL[IBX] * qL[IBX] + B2L / 2)) + epsilon;
  real_t caR       = sqrt(qR[IR] * (qR[IBX] * qR[IBX] + B2R / 2)) + epsilon;
  real_t cbL       = sqrt(qL[IR] * (qL[IR] * csL * csL + qL[IBY] * qL[IBY] + qL[IBZ] * qL[IBZ] + B2L / 2));
  real_t cbR       = sqrt(qR[IR] * (qR[IR] * csR * csR + qR[IBY] * qR[IBY] + qR[IBZ] * qR[IBZ] + B2R / 2));

  auto computeFastMagnetoAcousticSpeed = [&](const State &q, const real_t B2, const real_t cs)
  {
    const real_t c02  = cs * cs;
    const real_t ca2  = B2 / q[IR];
    const real_t cap2 = q[IBX] * q[IBX] / q[IR];
    return sqrt(0.5 * (c02 + ca2) + 0.5 * sqrt((c02 + ca2) * (c02 + ca2) - 4.0 * c02 * cap2));
  };

  // Using 3-wave if hyperbolicity is lost (from Dyablo)
  if (qL[IBX] * qR[IBX] < -epsilon || qL[IBY] * qR[IBY] < -epsilon || qL[IBZ] * qR[IBZ] < -epsilon)
  {
    const real_t cL = qL[IR] * computeFastMagnetoAcousticSpeed(qL, B2L, csL);
    const real_t cR = qR[IR] * computeFastMagnetoAcousticSpeed(qR, B2R, csR);
    const real_t c  = fmax(cL, cR);

    caL = c;
    caR = c;
    cbL = c;
    cbR = c;
  }

  const Vect cL{cbL, caL, caL};
  const Vect cR{cbR, caR, caR};

  // // 2. Compute star zone
  const Vect vL{qL[IU], qL[IV], qL[IW]};
  const Vect vR{qR[IU], qR[IV], qR[IW]};

  Vect Ustar{}, Pstar{};
  for (size_t i = 0; i < 3; ++i)
  {
    Ustar[i] = (cL[i] * vL[i] + cR[i] * vR[i] + pL[i] - pR[i]) / (cL[i] + cR[i]);
    Pstar[i] = (cR[i] * pL[i] + cL[i] * pR[i] + cL[i] * cR[i] * (vL[i] - vR[i])) / (cL[i] + cR[i]);
  }

  State q{};
  real_t Bstar;
  if (Ustar[IX] > 0.0)
  {
    q     = qL;
    Bstar = qR[IBX];
    pout  = qR[IP];
  }
  else
  {
    q     = qR;
    Bstar = qL[IBX];
    pout  = qL[IP];
  }
  const real_t beta_min      = 1.0e-3;
  const real_t alfven_max    = 10.0;
  const real_t beta          = q[IP] / (0.5 * (q[IBX] * q[IBX] + q[IBY] * q[IBY] + q[IBZ] * q[IBZ]));
  const real_t alfven_number = Kokkos::sqrt(q[IR] * q[IU] / (q[IBX] * q[IBX] + q[IBY] * q[IBY] + q[IBZ] * q[IBZ]));
  bool is_low_beta           = (beta < beta_min);
  State u                    = primToCons(q, params);
  // 3. Commpute flux
  real_t uS = Ustar[IX];
  flux[IR]  = u[IR] * uS;
  flux[IU]  = u[IU] * uS + Pstar[IX];
  flux[IV]  = u[IV] * uS + Pstar[IY];
  flux[IW]  = u[IW] * uS + Pstar[IZ];
  flux[IE]  = u[IE] * uS + Pstar[IX] * uS + Pstar[IY] * Ustar[IY] + Pstar[IZ] * Ustar[IZ];
  if (is_low_beta || (alfven_number > alfven_max))
  {
    flux[IBX] = u[IBX] * uS - q[IBX] * Ustar[IX];
    flux[IBY] = u[IBY] * uS - q[IBX] * Ustar[IY];
    flux[IBZ] = u[IBZ] * uS - q[IBX] * Ustar[IZ];
  }
  else
  {
    flux[IBX] = u[IBX] * uS - Bstar * Ustar[IX];
    flux[IBY] = u[IBY] * uS - Bstar * Ustar[IY];
    flux[IBZ] = u[IBZ] * uS - Bstar * Ustar[IZ];
  }
  flux[IPSI] = 0.0;
  // pout = Pstar[IX];
}
} // namespace fv2d

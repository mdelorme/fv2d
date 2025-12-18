#pragma once

namespace fv2d
{

/**
 * @brief Flux Splitting Lagrange Projection Solver
 *
 * FSLP solver as defined in Bourgeois et al. 2024 : "Recasting an operator splitting solver into a
 * standard finite volume flux-based algorithm. The case of a Lagrange-projection-type method for gas
 * dynamics", Journal of Computational Physics, vol 496.
 */
KOKKOS_INLINE_FUNCTION
void fslp(const State &qL, const State &qR, State &flux, real_t &pout, real_t gdx, const DeviceParams &params)
{
  // 1. Basic quantities
  const real_t rhoL = qL[IR];
  const real_t uL   = qL[IU];
  const real_t vL   = qL[IV];
  const real_t pL   = qL[IP];
  const real_t csL  = sqrt(params.gamma0 * pL / rhoL);

  const real_t rhoR = qR[IR];
  const real_t uR   = qR[IU];
  const real_t vR   = qR[IV];
  const real_t pR   = qR[IP];
  const real_t csR  = sqrt(params.gamma0 * pR / rhoR);

  // 2. Calculating theta and a
  const real_t ai    = params.fslp_K * Kokkos::max(rhoL * csL, rhoR * csR);                         // eq (17)
  const real_t theta = Kokkos::min(1.0, Kokkos::max(Kokkos::abs(uL) / csL, Kokkos::abs(uR) / csR)); // eq (77)

  // 3. Calculating u* and PI*
  //                                                  vvv this minus comes from g = -grad phi
  const real_t ustar = 0.5 * (uR + uL) - 0.5 / ai * (pR - pL - 0.5 * (rhoL + rhoR) * gdx); // eq (15)
  const real_t Pi    = 0.5 * (pR + pL) - theta * 0.5 * ai * (uR - uL);                     // eq (15)

  // 4. Upwinding
  const State &qstar  = (ustar > 0 ? qL : qR); // eq (32)
  const real_t Ekstar = 0.5 * qstar[IR] * (qstar[IU] * qstar[IU] + qstar[IV] * qstar[IV]);
  const real_t Estar  = Ekstar + qstar[IP] / (params.gamma0 - 1.0);

  // 5. Calculating flux : eq (31)
  flux[IR] = ustar * qstar[IR];
  flux[IU] = ustar * qstar[IR] * qstar[IU] + Pi;
  flux[IV] = ustar * qstar[IR] * qstar[IV];
  flux[IE] = ustar * (Estar + Pi);
}
} // namespace fv2d

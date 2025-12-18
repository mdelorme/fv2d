#pragma once

namespace fv2d
{
KOKKOS_INLINE_FUNCTION
void hll(State &qL, State &qR, State &flux, real_t &pout, const DeviceParams &params)
{
  const real_t aL = speedOfSound(qL, params);
  const real_t aR = speedOfSound(qR, params);

  // Davis' estimates for the signal speed
  const real_t sminL = qL[IU] - aL;
  const real_t smaxL = qL[IU] + aL;
  const real_t sminR = qR[IU] - aR;
  const real_t smaxR = qR[IU] + aR;

  const real_t SL = fmin(sminL, sminR);
  const real_t SR = fmax(smaxL, smaxR);

  auto computeFlux = [&](const State &q, const DeviceParams &params)
  {
    const real_t Ek = 0.5 * q[IR] * (q[IU] * q[IU] + q[IV] * q[IV]);
    const real_t E  = (q[IP] / (params.gamma0 - 1.0) + Ek);

    State fout{q[IR] * q[IU], q[IR] * q[IU] * q[IU] + q[IP], q[IR] * q[IU] * q[IV], (q[IP] + E) * q[IU]};

    return fout;
  };

  State FL = computeFlux(qL, params);
  State FR = computeFlux(qR, params);

  if (SL >= 0.0)
  {
    flux = FL;
    pout = qL[IP];
  }
  else if (SR <= 0.0)
  {
    flux = FR;
    pout = qR[IP];
  }
  else
  {
    State uL = primToCons(qL, params);
    State uR = primToCons(qR, params);
    pout     = 0.5 * (qL[IP] + qR[IP]);
    flux     = (SR * FL - SL * FR + SL * SR * (uR - uL)) / (SR - SL);
  }
}
} // namespace fv2d

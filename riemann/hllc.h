#pragma once

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void hllc(const State &qL, const State &qR, State &flux, real_t &pout, const DeviceParams &params)
{
  const real_t rL = qL[IR];
  const real_t uL = qL[IU];
  const real_t vL = qL[IV];
  const real_t pL = qL[IP];

  const real_t rR = qR[IR];
  const real_t uR = qR[IU];
  const real_t vR = qR[IV];
  const real_t pR = qR[IP];

  const real_t entho = 1.0 / (params.gamma0 - 1.0);

  const real_t ekL = 0.5 * rL * (uL * uL + vL * vL);
  const real_t EL  = ekL + pL * entho;
  const real_t ekR = 0.5 * rR * (uR * uR + vR * vR);
  const real_t ER  = ekR + pR * entho;

  const real_t cfastL = speedOfSound(qL, params);
  const real_t cfastR = speedOfSound(qR, params);

  const real_t SL = fmin(uL, uR) - fmax(cfastL, cfastR);
  const real_t SR = fmax(uL, uR) + fmax(cfastL, cfastR);

  const real_t rcL = rL * (uL - SL);
  const real_t rcR = rR * (SR - uR);

  const real_t uS = (rcR * uR + rcL * uL + (pL - pR)) / (rcR + rcL);
  const real_t pS = (rcR * pL + rcL * pR + rcL * rcR * (uL - uR)) / (rcR + rcL);

  const real_t rSL = rL * (SL - uL) / (SL - uS);
  const real_t ESL = ((SL - uL) * EL - pL * uL + pS * uS) / (SL - uS);

  const real_t rSR = rR * (SR - uR) / (SR - uS);
  const real_t ESR = ((SR - uR) * ER - pR * uR + pS * uS) / (SR - uS);

  State st;
  real_t E;
  if (SL > 0.0)
  {
    st   = qL;
    E    = EL;
    pout = pL;
  }
  else if (uS > 0.0)
  {
    st[IR] = rSL;
    st[IU] = uS;
    st[IV] = qL[IV];
    st[IP] = pS;
    E      = ESL;
    pout   = pS;
  }
  else if (SR > 0.0)
  {
    st[IR] = rSR;
    st[IU] = uS;
    st[IV] = qR[IV];
    st[IP] = pS;
    E      = ESR;
    pout   = pS;
  }
  else
  {
    st   = qR;
    E    = ER;
    pout = pR;
  }

  flux[IR] = st[IR] * st[IU];
  flux[IU] = st[IR] * st[IU] * st[IU] + st[IP];
  flux[IV] = flux[IR] * st[IV];
  flux[IE] = (E + st[IP]) * st[IU];
}

} // namespace fv2d

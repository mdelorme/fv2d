#pragma once

namespace fv2d
{
struct RiemannData
{
  real_t pout;
  real_t gdx;
};
} // namespace fv2d

#include "riemann/hll.h"
#include "riemann/hllc.h"
#include "riemann/fslp.h"

namespace fv2d
{

KOKKOS_INLINE_FUNCTION
void riemann(State &qL, State &qR, State &flux, RiemannData &rdata, const DeviceParams &params)
{
  switch (params.riemann_solver)
  {
  case HLL:
    hll(qL, qR, flux, rdata, params);
    break;
  case FSLP:
    fslp(qL, qR, flux, rdata, params);
    break;
  default:
    hllc(qL, qR, flux, rdata, params);
    break;
  }
}
} // namespace fv2d

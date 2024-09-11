#pragma once

#include "SimInfo.h"
#include "polyfit.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t GetGravityValue(Array Q, real_t y, const Params &params)
{
  const real_t g = params.g;
  switch(params.gravity_type){
    case GT_CONSTANT:
      return g;
      
    case GT_POLY_SINC:
      return 2 * g * (sin(y) - y*cos(y)) / (y*y);

    case GT_POLYFIT: 
      return get_g(y);

    default:
      return 0.0;
  }
}

}
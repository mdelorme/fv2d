#pragma once

#ifdef MHD
#include "StatesMHD.h"
constexpr bool mhd_run = true;
#else
#include "StatesHydro.h"
constexpr bool mhd_run = false;
#endif

}
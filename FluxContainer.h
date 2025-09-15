#pragma once

#include "States.h"

namespace fv2d {

struct Flux {
    State hydro;  // État hydrodynamique
    State mhd;    // État MHD

    Flux() : hydro(), mhd() {}

    Flux(const State& hydro_init, const State& mhd_init)
        : hydro(hydro_init), mhd(mhd_init) {}
};

} // namespace fv2d
#pragma once

#include <fstream>

#include "BoundaryConditions.h"
#include "SimInfo.h"

#include "init/InitData.h"

#include "init/hydro/Blast.h"
#include "init/hydro/C91.h"
#include "init/hydro/Diffusion.h"
#include "init/hydro/GreshoVortex.h"
#include "init/hydro/H84.h"
#include "init/hydro/KelvinHelmholtz.h"
#include "init/hydro/RayleighTaylor.h"
#include "init/hydro/Sod.h"

namespace fv2d
{

struct InitFunctor
{
private:
  Params full_params;

public:
  InitFunctor(Params &full_params) : full_params(full_params) {};
  ~InitFunctor() = default;

  void init(Array Q)
  {
    auto params       = full_params.device_params;
    auto init_problem = InitRegistry::getFunction(full_params.problem);
    InitData init_data(full_params);

    // Filling active domain ...
    Kokkos::parallel_for(
        "Initialization",
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) { init_problem(Q, i, j, params, init_data); });

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};

} // namespace fv2d

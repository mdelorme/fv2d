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

#include "init/InitFactory.h"

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
    // Filling domain
    auto formula = InitFactory::instantiate(full_params.problem);
    formula->init(Q, full_params);

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};

} // namespace fv2d

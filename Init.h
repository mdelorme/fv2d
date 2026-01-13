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
/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType
{
  SOD_X,
  SOD_Y,
  BLAST,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  C91,
  KELVIN_HELMHOLTZ,
  GRESHO_VORTEX
};

struct InitFunctor
{
private:
  Params full_params;
  InitType init_type;
  std::unique_ptr<InitFormula> formula;

public:
  InitFunctor(Params &full_params) : full_params(full_params)
  {
    std::map<std::string, InitType> init_map{{"sod_x", SOD_X},
                                             {"sod_y", SOD_Y},
                                             {"blast", BLAST},
                                             {"rayleigh-taylor", RAYLEIGH_TAYLOR},
                                             {"diffusion", DIFFUSION},
                                             {"H84", H84},
                                             {"C91", C91},
                                             {"kelvin_helmholtz", KELVIN_HELMHOLTZ},
                                             {"gresho_vortex", GRESHO_VORTEX}};

    if (init_map.count(full_params.problem) == 0)
      throw std::runtime_error("Error unknown problem " + full_params.problem);

    init_type = init_map[full_params.problem];
    switch (init_type) {
      case SOD_X:            formula = std::make_unique<InitSodX>();            break;
      case SOD_Y:            formula = std::make_unique<InitSodY>();            break;
      case BLAST:            formula = std::make_unique<InitBlast>();           break;
      case RAYLEIGH_TAYLOR:  formula = std::make_unique<InitRayleighTaylor>();  break;
      case DIFFUSION:        formula = std::make_unique<InitDiffusion>();       break;
      case H84:              formula = std::make_unique<InitH84>();             break;
      case C91:              formula = std::make_unique<InitC91>();             break;
      case KELVIN_HELMHOLTZ: formula = std::make_unique<InitKelvinHelmholtz>(); break;
      case GRESHO_VORTEX:    formula = std::make_unique<InitGreshoVortex>();    break;
    }
  };
  ~InitFunctor() = default;

  void init(Array Q)
  {
    // cppcheck-suppress shadowVariable
    formula->init(Q, full_params);

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};

} // namespace fv2d

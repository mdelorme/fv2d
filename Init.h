#pragma once

#include <fstream>

#include "BoundaryConditions.h"
#include "SimInfo.h"

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
  };
  ~InitFunctor() = default;

  void init(Array Q)
  {
    // cppcheck-suppress shadowVariable
    auto init_type = this->init_type;
    auto params    = full_params.device_params;

    RandomPool random_pool(full_params.seed);

    // Filling active domain ...
    Kokkos::parallel_for(
        "Initialization",
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          switch (init_type)
          {
          case SOD_X:
            initSodX(Q, i, j, params);
            break;
          case SOD_Y:
            initSodY(Q, i, j, params);
            break;
          case BLAST:
            initBlast(Q, i, j, params);
            break;
          case DIFFUSION:
            initDiffusion(Q, i, j, params);
            break;
          case RAYLEIGH_TAYLOR:
            initRayleighTaylor(Q, i, j, params);
            break;
          case H84:
            initH84(Q, i, j, params, random_pool);
            break;
          case C91:
            initC91(Q, i, j, params, random_pool);
            break;
          case KELVIN_HELMHOLTZ:
            initKelvinHelmholtz(Q, i, j, params);
            break;
          case GRESHO_VORTEX:
            initGreshoVortex(Q, i, j, params);
            break;
          }
        });

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};

} // namespace fv2d

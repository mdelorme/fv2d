#pragma once

#include <fstream>

#include "BoundaryConditions.h"
#include "SimInfo.h"

#include "init/hydro/blast.h"
#include "init/hydro/C91.h"
#include "init/hydro/diffusion.h"
#include "init/hydro/GreshoVortex.h"
#include "init/hydro/H84.h"
#include "init/hydro/KelvinHelmholtz.h"
#include "init/hydro/RayleighTaylor.h"
#include "init/hydro/sod.h"

#ifdef MHD
#include "init/mhd/ArtificialNonZeroDivB.h"
#include "init/mhd/Blast.h"
#include "init/mhd/BrioWu.h"
#include "init/mhd/DaiWoodward.h"
#include "init/mhd/Expansion.h"
#include "init/mhd/FieldLoopAdvection.h"
#include "init/mhd/KelvinHelmholtz.h"
#include "init/mhd/MagneticShearing.h"
#include "init/mhd/OrszagTang.h"
#include "init/mhd/RotatedShockTube.h"
#include "init/mhd/Rotor.h"
#include "init/mhd/SlowRarefaction.h"
#include "init/mhd/ShuOsher.h"
#endif // MHD

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
  GRESHO_VORTEX,
#ifdef MHD
  MHD_SOD_X,
  MHD_SOD_Y,
  ORSZAG_TANG,
  MHD_KELVIN_HELMHOLTZ,
  DAI_WOODWARD,
  BRIO_WU2,
  SLOW_RAREFACTION,
  EXPANSION1,
  EXPANSION2,
  SHU_OSHER,
  ARTIFICIAL_NON_ZERO_DIVB,
  BLAST_MHD_STANDARD,
  BLAST_MHD_LOW_BETA,
  ROTATED_SHOCK_TUBE,
  MHD_ROTOR,
  FIELD_LOOP_ADVECTION,
  SHEAR_B
#endif // MHD
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
                                             {"gresho_vortex", GRESHO_VORTEX},
#ifdef MHD
                                             {"mhd_sod_x", MHD_SOD_X},
                                             {"mhd_sod_y", MHD_SOD_Y},
                                             {"orszag-tang", ORSZAG_TANG},
                                             {"mhd_kelvin-helmholtz", MHD_KELVIN_HELMHOLTZ},
                                             {"dai-woodward", DAI_WOODWARD},
                                             {"brio-wu2", BRIO_WU2},
                                             {"slow-rarefaction", SLOW_RAREFACTION},
                                             {"expansion1", EXPANSION1},
                                             {"expansion2", EXPANSION2},
                                             {"shu-osher", SHU_OSHER},
                                             {"artifical_non_zero_divB", ARTIFICIAL_NON_ZERO_DIVB},
                                             {"blast_mhd_standard", BLAST_MHD_STANDARD},
                                             {"blast_mhd_low_beta", BLAST_MHD_LOW_BETA},
                                             {"rotated_shock_tube", ROTATED_SHOCK_TUBE},
                                             {"mhd_rotor", MHD_ROTOR},
                                             {"field_loop_advection", FIELD_LOOP_ADVECTION},
                                             {"shear_b", SHEAR_B}
#endif // MHD
    };

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
#ifdef MHD
          case MHD_SOD_X:
            initBrioWu1X(Q, i, j, params);
            break;
          case MHD_SOD_Y:
            initBrioWu1Y(Q, i, j, params);
            break;
          case ORSZAG_TANG:
            initOrszagTang(Q, i, j, params);
            break;
          case MHD_KELVIN_HELMHOLTZ:
            initMHDKelvinHelmholtz(Q, i, j, params, random_pool);
            break;
          case DAI_WOODWARD:
            initDaiWoodward(Q, i, j, params);
            break;
          case BRIO_WU2:
            initBrioWu2(Q, i, j, params);
            break;
          case SLOW_RAREFACTION:
            initSlowRarefaction(Q, i, j, params);
            break;
          case EXPANSION1:
            initExpansion1(Q, i, j, params);
            break;
          case EXPANSION2:
            initExpansion2(Q, i, j, params);
            break;
          case SHU_OSHER:
            initShuOsher(Q, i, j, params);
            break;
          case ARTIFICIAL_NON_ZERO_DIVB:
            initArtificialNonZeroDivB(Q, i, j, params);
            break;
          case BLAST_MHD_STANDARD:
            initBlastMHDStandard(Q, i, j, params);
            break;
          case BLAST_MHD_LOW_BETA:
            initBlastMHDLowBeta(Q, i, j, params);
            break;
          case ROTATED_SHOCK_TUBE:
            initRotatedShockTube(Q, i, j, params);
            break;
          case MHD_ROTOR:
            initMHDRotor(Q, i, j, params);
            break;
          case FIELD_LOOP_ADVECTION:
            initFieldLoopAdvection(Q, i, j, params);
            break;
          case SHEAR_B:
            shearB(Q, i, j, params);
            break;
#endif // MHD
          }
        });

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }

#ifdef MHD
  real_t initGLMch(Array Q, const Params &full_params) const
  {
    // > Calculate the time step for the GLM wave system, assuming ch=1.
    // > This is computed only once, then ch is computed frm the current timestep:
    // > dt~1/ch -> dt/dtch1 = 1/ch -> ch=dtch1/dt
    auto params     = full_params.device_params;
    real_t lambda_x = 0.0;
    real_t lambda_y = 0.0;
    Kokkos::parallel_reduce(
        "Compute inital GLM Wave Speed",
        full_params.range_dom,
        KOKKOS_LAMBDA(int i, int j, real_t &lambda_x, real_t &lambda_y) {
          State q = getStateFromArray(Q, i, j);
          // real_t bx          = q[IBX];
          // real_t by          = q[IBY];
          // real_t cs          = speedOfSound(q, params);
          // real_t va          = Kokkos::sqrt((bx * bx + by * by) / (4 * M_PI * q[IR]));
          // real_t v_fast      = Kokkos::sqrt(0.5 * (va * va + cs * cs +
          // Kokkos::sqrt((va * va + cs * cs) * (va * va + cs * cs) -
          //  4.0 * va * va * cs * cs * (bx * bx / (bx * bx + by * by)))));
          real_t lambdaloc_x = Kokkos::abs(q[IU] + fastMagnetoAcousticSpeed(q, params, IX));
          real_t lambdaloc_y = Kokkos::abs(q[IV] + fastMagnetoAcousticSpeed(q, params, IY));
          lambda_x           = Kokkos::max(lambda_x, lambdaloc_x);
          lambda_y           = Kokkos::max(lambda_y, lambdaloc_y);
        },
        Kokkos::Max<real_t>(lambda_x),
        Kokkos::Max<real_t>(lambda_y));
    return params.CFL * 2.0 / (lambda_x / params.dx + lambda_y / params.dy);
  }
#endif // MHD
};

} // namespace fv2d

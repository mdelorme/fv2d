#pragma once

#include <fstream>
#include <Kokkos_Random.hpp>

#include "SimInfo.h"
#include "BoundaryConditions.h"

namespace fv2d {

namespace {

  using RandomPool = Kokkos::Random_XorShift64_Pool<>;

  /**
   * @brief Sod Shock tube aligned along the X axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodX(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IX] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IU) = 0.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IU) = 0.0;
    }
  }

  /**
   * @brief Sod Shock tube aligned along the Y axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodY(Array Q, int i, int j, const DeviceParams &params) {
    if (getPos(params, i, j)[IY] <= 0.5) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IP) = 1.0;
      Q(j, i, IU) = 0.0;
    }
    else {
      Q(j, i, IR) = 0.125;
      Q(j, i, IP) = 0.1;
      Q(j, i, IU) = 0.0;
    }
  }

  /**
   * @brief Sedov blast initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t xr = xmid - x;
    real_t yr = ymid - y;
    real_t r = sqrt(xr*xr+yr*yr);

    if (r < 0.2) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 10.0;
    }
    else {
      Q(j, i, IR) = 1.2;
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = 0.1;
    }
  }

  /**
   * @brief Stratified convection based on Hurlburt et al 1984
   */
  KOKKOS_INLINE_FUNCTION
  void initH84(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t rho = pow(y, params.m1);
    real_t prs = pow(y, params.m1+1.0); 

    auto generator = random_pool.get_state();
    real_t pert = params.h84_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    Q(j, i, IR) = rho;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = pert;
    Q(j, i, IP) = prs;
  }

  /**
   * @brief Stratified convection based on Cattaneo et al. 1991
   */
  KOKKOS_INLINE_FUNCTION
  void initC91(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t T = (1.0 + params.theta1*y);
    real_t rho = pow(T, params.m1);
    real_t prs = pow(T, params.m1+1.0);

    auto generator = random_pool.get_state();
    real_t pert = params.c91_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    prs = prs * (1.0 + pert);

    Q(j, i, IR) = rho;
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IP) = prs;
  }

  /**
   * @brief Simple diffusion test with a structure being advected on the grid
   */
  KOKKOS_INLINE_FUNCTION
  void initDiffusion(Array Q, int i, int j, const DeviceParams &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = getPos(params, i, j);

    real_t x0 = (pos[IX]-xmid);
    real_t y0 = (pos[IY]-ymid);

    real_t r = sqrt(x0*x0+y0*y0);

    if (r < 0.2) 
      Q(j, i, IR) = 1.0;
    else
      Q(j, i, IR) = 0.1;

    Q(j, i, IP) = 1.0;
    Q(j, i, IU) = 1.0;
    Q(j, i, IV) = 1.0;
  }

  /**
   * @brief Rayleigh-Taylor instability setup
   */
  KOKKOS_INLINE_FUNCTION
  void initRayleighTaylor(Array Q, int i, int j, const DeviceParams &params) {
    real_t ymid = 0.5*(params.ymin + params.ymax);

    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    const real_t P0 = 2.5;

    if (y < ymid) {
      Q(j, i, IR) = 1.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = P0 + 0.1 * params.gy * y;
    }
    else {
      Q(j, i, IR) = 2.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = P0 + 0.1 * params.gy * y;
    }

    if (y > -1.0/3.0 && y < 1.0/3.0)
      Q(j, i, IV) = 0.01 * (1.0 + cos(4*M_PI*x)) * (1 + cos(3.0*M_PI*y))/4.0;
  }

  /**
   * @brief Kelvin-Helmholtz instability setup
   * 
   * Taken from Lecoanet et al, "A validated non-linear Kelvin–Helmholtz benchmark for numerical hydrodynamics"
   * 2016, MNRAS
   */
  KOKKOS_INLINE_FUNCTION
  void initKelvinHelmholtz(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    const real_t q1 = Kokkos::tanh((y-params.kh_y1)/params.kh_a);
    const real_t q2 = Kokkos::tanh((y-params.kh_y2)/params.kh_a);
    const real_t s2 = params.kh_sigma*params.kh_sigma;
    const real_t dy1 = (y-params.kh_y1)*(y-params.kh_y1);
    const real_t dy2 = (y-params.kh_y2)*(y-params.kh_y2);
    const real_t rho = 1.0 + params.kh_rho_fac*0.5*(q1-q2);
    const real_t u   = params.kh_uflow * (q1-q2-1.0);
    const real_t v   = params.kh_amp * Kokkos::sin(2.0*M_PI*x)*(Kokkos::exp(-dy1/s2) + Kokkos::exp(-dy2/s2));
    
    Q(j, i, IR) = rho;
    Q(j, i, IU) = u;
    Q(j, i, IV) = v;
    Q(j, i, IP) = params.kh_P0;
  }

  /**
   * @brief Gresho-Vortex setup for Low-mach flows
   * 
   * Based on Miczek et al. 2015 "New numerical solver for flows at various Mach numbers"
   */
  KOKKOS_INLINE_FUNCTION
  void initGreshoVortex(Array Q, int i, int j, const DeviceParams &params) {
    Pos pos = getPos(params, i, j);
    const real_t xmid = 0.5*(params.xmin + params.xmax);
    const real_t ymid = 0.5*(params.ymin + params.ymax);
    const real_t xr = pos[IX]-xmid;
    const real_t yr = pos[IY]-ymid;
    const real_t r = sqrt(xr*xr+yr*yr);

    // Pressure is given from density and Mach
    const real_t pressure = params.gresho_density / (params.gamma0 * params.gresho_Mach*params.gresho_Mach);

    Q(j, i, IR) = params.gresho_density;

    real_t u_phi;
    if (r < 0.2) {
      u_phi = 5.0*r;
      Q(j, i, IP) = pressure + 12.5*r*r;
    }
    else if (r < 0.4) {
      u_phi = 2.0 - 5.0*r;
      Q(j, i, IP) = pressure + 12.5*r*r + 4.0*(1.0-5.0*r+log(5.0*r));
    }
    else {
      u_phi = 0.0;
      Q(j, i, IP) = pressure - 2.0 + 4.0*log(2.0);
    }

    const real_t xnr = xr / r;
    const real_t ynr = yr / r;
    Q(j, i, IU) = -ynr * u_phi;
    Q(j, i, IV) =  xnr * u_phi;
    
  }
}



/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType {
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

struct InitFunctor {
private:
  Params full_params;
  InitType init_type;
public:
  InitFunctor(Params &full_params)
    : full_params(full_params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
      {"sod_y", SOD_Y},
      {"blast", BLAST},
      {"rayleigh-taylor", RAYLEIGH_TAYLOR},
      {"diffusion", DIFFUSION},
      {"H84", H84},
      {"C91", C91},
      {"kelvin_helmholtz", KELVIN_HELMHOLTZ},
      {"gresho_vortex", GRESHO_VORTEX},
    };

    if (init_map.count(full_params.problem) == 0)
      throw std::runtime_error("Error unknown problem " + full_params.problem);

    init_type = init_map[full_params.problem];
  };
  ~InitFunctor() = default;

  void init(Array &Q) {
    auto init_type = this->init_type;
    auto params = full_params.device_params;

    RandomPool random_pool(full_params.seed);

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          full_params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j) {
                            switch(init_type) {
                              case SOD_X:            initSodX(Q, i, j, params); break;
                              case SOD_Y:            initSodY(Q, i, j, params); break;
                              case BLAST:            initBlast(Q, i, j, params); break;
                              case DIFFUSION:        initDiffusion(Q, i, j, params); break;
                              case RAYLEIGH_TAYLOR:  initRayleighTaylor(Q, i, j, params); break;
                              case H84:              initH84(Q, i, j, params, random_pool); break;
                              case C91:              initC91(Q, i, j, params, random_pool); break;
                              case KELVIN_HELMHOLTZ: initKelvinHelmholtz(Q, i, j, params); break;
                              case GRESHO_VORTEX:   initGreshoVortex(Q, i, j, params); break;
                            }
                          });
  
    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};



}
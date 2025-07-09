#pragma once

#include <fstream>
#include <Kokkos_Random.hpp>

#include "SimInfo.h"
#include "BoundaryConditions.h"

#include "Geometry.h"

namespace fv2d {

namespace {

  using RandomPool = Kokkos::Random_XorShift64_Pool<>;

  /**
   * @brief Sod Shock tube aligned along the X axis
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initSodX(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
    Pos pos = geometry.map_to_physical_center(i, j);
    if (pos[IX] <= 0.5) {
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initSodY(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
    Pos pos = geometry.map_to_physical_center(i, j);
    if (pos[IY] <= 0.5) {
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initRiemann2D(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
    auto [x, y] = geometry.map_to_physical_center(i, j);
    int qid = (x < 0.5) + 2*(y < 0.5);

    State quadrants[4] = {
      {1.5,    0.0,   0.0,   1.5},  // tr
      {0.5323, 1.206, 0.0,   0.3},  // tl
      {0.5323, 0.0,   1.206, 0.3},  // bl
      {0.138,  1.206, 1.206, 0.029} // br
    };
    setStateInArray(Q, i, j, quadrants[qid]);
  }

  /**
   * @brief Sedov blast initial conditions
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = geometry.map_to_physical_center(i,j);
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param random_pool random number generator
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initH84(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool, const Geometry &geometry) {
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param random_pool random number generator
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initC91(Array Q, int i, int j, const DeviceParams &params, const RandomPool &random_pool, const Geometry &geometry) {
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initDiffusion(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
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
   * 
   * @param Q array of primitive variables
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param params global parameters of the run
   * @param geometry geometry object for position
   */
  KOKKOS_INLINE_FUNCTION
  void initRayleighTaylor(Array Q, int i, int j, const DeviceParams &params, const Geometry &geometry) {
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
}



/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType {
  SOD_X,
  SOD_Y,
  BLAST,
  RIEMANN_2D,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  C91
};

struct InitFunctor {
private:
  Params full_params;
  Geometry geometry;
  InitType init_type;
public:
  InitFunctor(Params &full_params)
    : full_params(full_params),
      geometry(full_params.device_params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
      {"sod_y", SOD_Y},
      {"blast", BLAST},
      {"riemann_2D", RIEMANN_2D},
      {"rayleigh-taylor", RAYLEIGH_TAYLOR},
      {"diffusion", DIFFUSION},
      {"H84", H84},
      {"C91", C91},
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

    auto &geometry = this->geometry;

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          full_params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j) {
                            switch(init_type) {
                              case SOD_X:           initSodX(Q, i, j, params, geometry); break;
                              case SOD_Y:           initSodY(Q, i, j, params, geometry); break;
                              case BLAST:           initBlast(Q, i, j, params, geometry); break;
                              case RIEMANN_2D:      initRiemann2D(Q, i, j, params, geometry); break;
                              case DIFFUSION:       initDiffusion(Q, i, j, params, geometry); break;
                              case RAYLEIGH_TAYLOR: initRayleighTaylor(Q, i, j, params, geometry); break;
                              case H84:             initH84(Q, i, j, params, random_pool, geometry); break;
                              case C91:             initC91(Q, i, j, params, random_pool, geometry); break;
                            }
                          });
  
    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};



}
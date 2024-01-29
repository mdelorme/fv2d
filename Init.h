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
      Q(j, i, IP) = P0 + 0.1 * params.g * y;
    }
    else {
      Q(j, i, IR) = 2.0;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = P0 + 0.1 * params.g * y;
    }

    if (y > -1.0/3.0 && y < 1.0/3.0)
      Q(j, i, IV) = 0.01 * (1.0 + cos(4*M_PI*x)) * (1 + cos(3.0*M_PI*y))/4.0;
  }

  /**
   * @brief Tri-Layer setup for a Currie2020 type of run
   *
   */
  KOKKOS_INLINE_FUNCTION
  void initTriLayer(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    const real_t y = pos[IY];

    const real_t T0 = params.T0;
    const real_t rho0 = params.rho0;
    const real_t p0 = rho0 * T0;

    const real_t T1   = T0 + params.theta2 * params.tri_y1;
    const real_t rho1 = rho0 * pow(T1/T0, params.m2);
    const real_t p1   = p0 * pow(T1/T0, params.m2+1.0);

    const real_t T2   = T1 + params.theta1 * (params.tri_y2-params.tri_y1);
    const real_t rho2 = rho1 * pow(T2/T1, params.m1);
    const real_t p2   = p1 * pow(T2/T1, params.m1+1.0);

    // Top layer
    real_t T;
    if (y <= params.tri_y1) {
      T = T0 + params.theta2*y;
      Q(j, i, IR) = rho0 * pow(T/T0, params.m2);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = p0 * pow(T/T0, params.m2+1.0);
    }
    // Middle layer
    else if (y <= params.tri_y2) {
      auto generator = random_pool.get_state();
      T = T1 + params.theta1*(y-params.tri_y1);
      Q(j, i, IR) = rho1 * pow(T/T1, params.m1);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;

      real_t pert = params.tri_pert * (generator.drand(-0.5, 0.5));
      if (y-params.tri_y1 < 0.1 || params.tri_y2-y < 0.1)
        pert = 0.0;
      Q(j, i, IP) = (p1 * pow(T/T1, params.m1+1.0)) * (1.0 + pert);
      random_pool.free_state(generator);
    }
    // Bottom layer
    else {
      T = T2 + params.theta2*(y-params.tri_y2);
      Q(j, i, IR) = rho2 * pow(T/T2, params.m2);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = p2 * pow(T/T2, params.m2+1.0);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void initTriLayerSmooth(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    const real_t y = pos[IY];

    const real_t T0 = params.T0;
    const real_t rho0 = params.rho0;
    const real_t p0 = 8.31 * rho0 * T0;

    const real_t T1   = T0 + params.tri_y1 / params.kappa / params.tri_k2;
    const real_t rho1 = rho0 * pow(T1/T0, params.m2);
    const real_t p1   = p0 * pow(T1/T0, params.m2+1.0);

    const real_t T2   = T1 + (params.tri_y2-params.tri_y1) / params.kappa / params.tri_k1;
    const real_t rho2 = rho1 * pow(T2/T1, params.m1);
    const real_t p2   = p1 * pow(T2/T1, params.m1+1.0);

    // Smooth temperature profile
    real_t T;
    real_t th = 0.1;
    if (y <= params.tri_y2 - (params.tri_y2-params.tri_y1)/2.) {
      real_t Tin = T0 + y / params.kappa / params.tri_k2;
      real_t Tout = T1 + (y-params.tri_y1) / params.kappa / params.tri_k1;
      real_t fin = (tanh((params.tri_y1-y)/th) + 1.0) * 0.5;
      real_t fout = (tanh((y-params.tri_y1)/th) + 1.0) * 0.5;
      T = Tin*fin+Tout*fout;
    }
    else {
      real_t Tin = T1 + (y-params.tri_y1) / params.kappa / params.tri_k1;
      real_t Tout = T2 + (y-params.tri_y2) / params.kappa / params.tri_k2;
      real_t fin = (tanh((params.tri_y2-y)/th) + 1.0) * 0.5;
      real_t fout = (tanh((y-params.tri_y2)/th) + 1.0) * 0.5;
      T = Tin*fin+Tout*fout;
    }
    // Top layer
    if (y <= params.tri_y1) {
      Q(j, i, IR) = rho0 * pow(T/T0, params.m2);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = p0 * pow(T/T0, params.m2+1.0);
    }
    // Middle layer
    else if (y <= params.tri_y2) {
      auto generator = random_pool.get_state();
      Q(j, i, IR) = rho1 * pow(T/T1, params.m1);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      real_t pert = params.tri_pert * (generator.drand(-0.5, 0.5));
      if (y-params.tri_y1 < 0.1 || params.tri_y2-y < 0.1)
        pert = 0.0;
      Q(j, i, IP) = (p1 * pow(T/T1, params.m1+1.0)) * (1.0 + pert);
      random_pool.free_state(generator);
    }
    // Bottom layer
    else {
      Q(j, i, IR) = rho2 * pow(T/T2, params.m2);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) = p2 * pow(T/T2, params.m2+1.0);
    }
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
  B02,
  TRI_LAYER,
  TRI_LAYER_SMOOTH
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
      {"tri-layer", TRI_LAYER},
      {"tri-layer-smooth", TRI_LAYER_SMOOTH}
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
                              case SOD_X:           initSodX(Q, i, j, params); break;
                              case SOD_Y:           initSodY(Q, i, j, params); break;
                              case BLAST:           initBlast(Q, i, j, params); break;
                              case DIFFUSION:       initDiffusion(Q, i, j, params); break;
                              case RAYLEIGH_TAYLOR: initRayleighTaylor(Q, i, j, params); break;
                              case H84:             initH84(Q, i, j, params, random_pool); break;
                              case C91:             initC91(Q, i, j, params, random_pool); break;
                              case TRI_LAYER:       initTriLayer(Q, i, j, params, random_pool); break;
                              case TRI_LAYER_SMOOTH:initTriLayerSmooth(Q, i, j, params, random_pool); break;
                              case B02:             break;
                              default: break;
                            }
                          });

    // ... and boundaries
    BoundaryManager bc(full_params);
    bc.fillBoundaries(Q);
  }
};



}

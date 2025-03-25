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
  void initSodX(Array Q, int i, int j, const Params &params) {
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
  void initSodY(Array Q, int i, int j, const Params &params) {
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
   * @brief Sod Shock tube aligned along the diagonal.
   */
  KOKKOS_INLINE_FUNCTION
  void initSod45(Array Q, int i, int j, const Params &params) {
    auto [x, y] = getPos(params, i, j);
    if (x > y) {
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
   * @brief Kelvin-Helmholtz
   */
  KOKKOS_INLINE_FUNCTION
  void initKelvinHelmholtz(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
    real_t y = getPos(params, i, j)[IY] - 0.5;
    bool inside = abs(y) < 0.25;

    auto generator = random_pool.get_state();
    real_t pert_u = 0.01 * (generator.drand(-0.5, 0.5));
    real_t pert_v = 0.01 * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    real_t rho = inside ? 2.0 : 1.0;
    real_t u   = inside ? 0.5 : -0.5;

    State q = {rho, u + pert_u, pert_v, 2.5};
    setStateInArray(Q, i, j, q);
  }

  /**
   * @brief Kelvin-Helmholtz radial
   */
  KOKKOS_INLINE_FUNCTION
  void initKelvinHelmholtzRadial(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
    auto [x, y] = getPos(params, i, j);
    x-=0.5; y-=0.5;
    real_t r = sqrt(x*x + y*y);
    const real_t nx = x/r;
    const real_t ny = y/r;

    constexpr real_t r0 = 0.25;
    constexpr real_t uflow = 0.5;
    constexpr real_t rho_in = 2.0;
    constexpr real_t rho_out = 1.0;
    constexpr real_t p0 = 2.5;
    const real_t velocity = uflow * r;

    auto generator = random_pool.get_state();
    real_t pert_u = 0.02 * r * (generator.drand(-0.5, 0.5));
    real_t pert_v = 0.02 * r * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    constexpr real_t thck = 0.0025;
    real_t tr = 0.5 * (tanh((r - r0) / thck) + 1.0);
    
    const State Qin  {rho_in,  (velocity *  ny + pert_u * nx), (velocity * -nx + pert_v * ny), p0 + 0.5 * uflow * uflow * rho_in * r*r};
    const State Qout {rho_out, (velocity * -ny + pert_u * nx), (velocity *  nx + pert_v * ny), p0 + 0.5 * uflow * uflow * (rho_out * (r*r-r0*r0) + rho_in * r0*r0)};

    State q = (1-tr) * Qin + tr * Qout;
    setStateInArray(Q, i, j, q);
  }

  /**
   * @brief Riemann2D
   */
  KOKKOS_INLINE_FUNCTION
  void initRiemann2D(Array Q, int i, int j, const Params &params) {
    auto [x, y] = getPos(params, i, j);
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
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, const Params &params) {
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
      Q(j, i, IP) = 10.0;
    }
    else {
      Q(j, i, IR) = 1.2;
      Q(j, i, IU) = 0.0;
      Q(j, i, IP) = 0.1;
    }
  }

  /**
   * @brief Sedov blast initial conditions with smoothing
   */
  KOKKOS_INLINE_FUNCTION
  void initSmoothBlast(Array Q, int i, int j, const Params &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t xr = xmid - x;
    real_t yr = ymid - y;
    real_t r = sqrt(xr*xr+yr*yr);

    constexpr real_t thck = 0.02;
    constexpr real_t rblast = 0.2;
    real_t tr = 0.5 * (tanh((r - rblast) / thck) + 1.0);

    const State Qin  {1.0, 0.0, 0.0, 10.0};
    const State Qout {1.2, 0.0, 0.0,  0.1};

    State res = (1-tr) * Qin + tr * Qout;
    setStateInArray(Q, i, j, res);
  }

  /**
   * @brief Stratified convection based on Hurlburt et al 1984
   */
  KOKKOS_INLINE_FUNCTION
  void initH84(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t T = y;
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
  void initC91(Array Q, int i, int j, const Params &params, const RandomPool &random_pool) {
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
  void initDiffusion(Array Q, int i, int j, const Params &params) {
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
  void initRayleighTaylor(Array Q, int i, int j, const Params &params) {
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
}



/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType {
  SOD_X,
  SOD_Y,
  SOD_45,
  KELVIN_HELM,
  KELVIN_HELM_R,
  RIEMANN2D,
  BLAST,
  SMOOTHBLAST,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  C91
};

struct InitFunctor {
private:
  Params params;
  InitType init_type;
public:
  InitFunctor(Params &params)
    : params(params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
      {"sod_y", SOD_Y},
      {"sod_45", SOD_45},
      {"kelvin-helmholtz", KELVIN_HELM},
      {"kelvin-helmholtz-radial", KELVIN_HELM_R},
      {"riemann2d", RIEMANN2D},
      {"blast", BLAST},
      {"smoothblast", SMOOTHBLAST},
      {"rayleigh-taylor", RAYLEIGH_TAYLOR},
      {"diffusion", DIFFUSION},
      {"H84", H84},
      {"C91", C91}
    };

    if (init_map.count(params.problem) == 0)
      throw std::runtime_error("Error unknown problem " + params.problem);

    init_type = init_map[params.problem];
  };
  ~InitFunctor() = default;

  void init(Array &Q) {
    auto init_type = this->init_type;
    auto params = this->params;

    RandomPool random_pool(params.seed);

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j) {
                            switch(init_type) {
                              case SOD_X:           initSodX(Q, i, j, params); break;
                              case SOD_Y:           initSodY(Q, i, j, params); break;
                              case SOD_45:          initSod45(Q, i, j, params); break;
                              case KELVIN_HELM:     initKelvinHelmholtz(Q, i, j, params, random_pool); break;
                              case KELVIN_HELM_R:   initKelvinHelmholtzRadial(Q, i, j, params, random_pool); break;
                              case RIEMANN2D:       initRiemann2D(Q, i, j, params); break;
                              case BLAST:           initBlast(Q, i, j, params); break;
                              case SMOOTHBLAST:     initSmoothBlast(Q, i, j, params); break;
                              case DIFFUSION:       initDiffusion(Q, i, j, params); break;
                              case RAYLEIGH_TAYLOR: initRayleighTaylor(Q, i, j, params); break;
                              case H84:             initH84(Q, i, j, params, random_pool); break;
                              case C91:             initC91(Q, i, j, params, random_pool); break;
                            }
                          });
  
    // ... and boundaries
    BoundaryManager bc(params);
    bc.fillBoundaries(Q);
  }
};



}
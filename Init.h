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
   */
  KOKKOS_INLINE_FUNCTION
  void initSodX(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    Pos pos = geo.mapc2p_center(i,j);
  
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
   */
  KOKKOS_INLINE_FUNCTION
  void initSodY(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    Pos pos = geo.mapc2p_center(i,j);

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
   * @brief Lax-Liu quadrants
   */
  KOKKOS_INLINE_FUNCTION
  void initLaxLiu(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    Pos pos = geo.mapc2p_center(i,j);

    int qid = (pos[IX] < 0.5) + 2*(pos[IY] < 0.5);

    switch(qid)
    {
      case 0: // tr
        Q(j, i, IR) = 1.5;
        Q(j, i, IU) = 0.0;
        Q(j, i, IV) = 0.0;
        Q(j, i, IP) = 1.5;
        break;
      case 1: // tl
        Q(j, i, IR) = 0.5323;
        Q(j, i, IU) = 1.206;
        Q(j, i, IV) = 0.0;
        Q(j, i, IP) = 0.3;
        break;
      case 2: // bl
        Q(j, i, IR) = 0.5323;
        Q(j, i, IU) = 0.0;
        Q(j, i, IV) = 1.206;
        Q(j, i, IP) = 0.3;
        break;
      case 3: // br
        Q(j, i, IR) = 0.138;
        Q(j, i, IU) = 1.206;
        Q(j, i, IV) = 1.206;
        Q(j, i, IP) = 0.029;
      default: break;
    }
  }

  /**
   * @brief Sedov blast initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);

    Pos pos = geo.mapc2p_center(i,j);

    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t xr = xmid - x;
    real_t yr = ymid - y;
    real_t r = sqrt(xr*xr+yr*yr);

    const real_t xsize = (params.xmax - params.xmin);
    const real_t ysize = (params.ymax - params.ymin);
    const real_t blast_radius = 0.20 * ((xsize < ysize) ? xsize : ysize);

    #if 0
      if (r < blast_radius) {
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

    #else // lissage
      const real_t thck = 0.02;
      real_t tr = 0.5 * (tanh((r - blast_radius) / thck) + 1.0);
      
      Q(j, i, IR) = 1.2 * tr + 1.0 * (1-tr);
      Q(j, i, IU) = 0.0;
      Q(j, i, IV) = 0.0;
      Q(j, i, IP) =  0.1 * tr + 10.0 * (1-tr);
    #endif
  }
//////////////////////////////////////////////////////////////////////////
////////////////////////////      RING       /////////////////////////////
//////////////////////////////////////////////////////////////////////////

  /**
   * @brief Sedov blast initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initRingBlast(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    Pos pos = geo.mapc2p_center(i,j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    constexpr real_t r0 = 0.75;
    constexpr real_t r_width = 0.05;
    real_t r = sqrt(x*x+y*y);

    constexpr real_t thck = 0.02;
    real_t tr = 0.5 * (tanh( (fabs(r - r0) - r_width) / thck) + 1.0);
    
    Q(j, i, IR) = 1.2 * tr + 1.0 * (1-tr);
    Q(j, i, IU) = 0.0;
    Q(j, i, IV) = 0.0;
    Q(j, i, IP) =  0.1 * tr + 10.0 * (1-tr);
  }

  KOKKOS_INLINE_FUNCTION
  void initRingInit(Array Q, int i, int j, const Params &params, const Geometry &geo) {
    Pos pos = geo.mapc2p_center(i,j);
    real_t x = pos[IX];
    real_t y = pos[IY];

    real_t r = sqrt(x*x + y*y);
    real_t cos = x / r;
    real_t sin = y / r;

    real_t velocity = params.ring_velocity * ((params.ring_scale_vel_r == true) ? r : 1.0);

    bool cond;
    if(params.ring_init_type == 1)
      cond = (r < 0.75);
    else
      cond = (fabs(r - 0.75) < params.init_type2_radius);

    // if(cond){
    //   Q(j, i, IR) = params.ring_rho_in;
    //   Q(j, i, IU) = velocity * sin;
    //   Q(j, i, IV) = velocity * -cos;
    //   Q(j, i, IP) = params.ring_p_in;
    // }
    // else{
    //   Q(j, i, IR) = params.ring_rho_out;
    //   Q(j, i, IU) = - velocity * sin;
    //   Q(j, i, IV) = - velocity * -cos;
    //   Q(j, i, IP) = params.ring_p_out;
    // }


    // fix pressure

    // TODO: pb sur la velocity, norme tjr = 1 ?!
    if(cond){
      const real_t fix_factor = params.ring_rho_in * params.ring_velocity * params.ring_velocity * log(r / 0.5);
      Q(j, i, IR) = params.ring_rho_in;
      Q(j, i, IU) = velocity * sin;
      Q(j, i, IV) = velocity * -cos;
      Q(j, i, IP) = params.ring_p_in + fix_factor;
    }
    else{
      const real_t fix_factor = params.ring_rho_out * params.ring_velocity * params.ring_velocity * log(r / 0.5);
      Q(j, i, IR) = params.ring_rho_out;
      Q(j, i, IU) = - velocity * sin;
      Q(j, i, IV) = - velocity * -cos;
      Q(j, i, IP) = params.ring_p_out + fix_factor;
    }
  }



//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
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
  BLAST,
  RING_BLAST,
  RING_INIT,

  LAXLIU,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  C91
};

struct InitFunctor {
private:
  Params params;
  InitType init_type;
  Geometry geometry;
public:
  InitFunctor(Params &params)
    : params(params),
      geometry(params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
      {"sod_y", SOD_Y},
      {"blast", BLAST},
      {"laxliu", LAXLIU},
      {"ring_blast", RING_BLAST},
      {"ring_init", RING_INIT},

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
    auto geometry = this->geometry;

    RandomPool random_pool(params.seed);

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j) {
                            switch(init_type) {
                              case SOD_X:           initSodX(Q, i, j, params, geometry); break;
                              case SOD_Y:           initSodY(Q, i, j, params, geometry); break;
                              case BLAST:           initBlast(Q, i, j, params, geometry); break;
                              case LAXLIU:          initLaxLiu(Q, i, j, params, geometry); break;
                              case RING_BLAST:      initRingBlast(Q, i, j, params, geometry); break;
                              case RING_INIT:       initRingInit(Q, i, j, params, geometry); break;
                              // case DIFFUSION:       initDiffusion(Q, i, j, params); break;
                              // case RAYLEIGH_TAYLOR: initRayleighTaylor(Q, i, j, params); break;
                              // case H84:             initH84(Q, i, j, params, random_pool); break;
                              // case C91:             initC91(Q, i, j, params, random_pool); break;
                            }
                          });
  
    // ... and boundaries
    BoundaryManager bc(params);
    bc.fillBoundaries(Q);
  }
};



}
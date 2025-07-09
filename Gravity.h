#pragma once

#include "SimInfo.h"
#include "Geometry.h"

namespace fv2d {

/**
 * @brief Computes the analytical gravity at a certain position for a certain direction
 * 
 * @param i, j indices of the current cell
 * @param params the parameters of the run
 */
KOKKOS_INLINE_FUNCTION
Pos getAnalyticalGravity(int i, int j, const DeviceParams &params) {
  Pos pos = getPos(params, i, j);
  Pos g;

  switch(params.analytical_gravity_mode) {
    case AGM_HOT_BUBBLE: 
    default: 
      g[IX] = 0.0;
      g[IY] = params.hot_bubble_g0 * Kokkos::sin(pos[IY] * M_PI * 2.0 / params.ymax); 
  }

  return g;
}

/**
 * @brief Method to compute the gravitational acceleration along a direction
 * 
 * @param i, j indices of the current cell
 * @param params the parameters of the run
 * @param geometry the geometry object
 */
KOKKOS_INLINE_FUNCTION
Pos getGravity(int i, int j, const DeviceParams &params, const Geometry &geometry) {
  Pos g;
  switch (params.gravity_mode) {
    case GRAV_CONSTANT: 
      g[IX] = params.gx;
      g[IY] = params.gy; 
      break;
    case GRAV_ANALYTICAL: g = getAnalyticalGravity(i, j, params); break;
    case GRAV_NONE:
    default: 
      g[IX] = 0.0; 
      g[IY] = 0.0;
  }

  return g;
}

/**
 * Functor applying gravity on the grid
 */
class GravityFunctor {
public:
  Params full_params;
  Geometry geometry;

  GravityFunctor(const Params &full_params) 
    : full_params(full_params), geometry(full_params.device_params) {};
  ~GravityFunctor() = default;

  void applyGravity(Array Q, Array Unew, real_t dt) {
    auto &full_params = this->full_params;
    auto &params      = full_params.device_params;
    auto &geometry    = this->geometry;

    Kokkos::parallel_for(
      "Gravity", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        Pos g = getGravity(i, j, params, geometry);

        real_t rhoOld = Q(j, i, IR);
        real_t rhoNew = Unew(j, i, IR);
        real_t rhou = Unew(j, i, IU);
        real_t rhov = Unew(j, i, IV);
        real_t ekin_old = 0.5 * (rhou*rhou + rhov*rhov) / rhoNew;
        rhou += 0.5 * dt * g[IX] * (rhoOld + rhoNew);
        rhov += 0.5 * dt * g[IY] * (rhoOld + rhoNew);
        Unew(j, i, IU) = rhou;
        Unew(j, i, IV) = rhov;
        real_t ekin_new = 0.5 * (rhou*rhou + rhov*rhov) / rhoNew;
        Unew(j, i, IE) += (ekin_new - ekin_old);
      });
  }
};



}
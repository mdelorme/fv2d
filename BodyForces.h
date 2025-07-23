#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
Pos computeGravity(int i, int j, const DeviceParams &params, const Geometry &geometry) {
  Pos g{0};

  switch (params.gravity_mode) {
    case GRAV_READFILE:
    {
      Pos pos = geometry.mapc2p_center(i,j);
      const real_t r = norm(pos);
      g = params.spl_grav.GetValue(r) / r * pos;
      break;
    }
    case GRAV_RADIAL_LIN:
      g = -params.gy * geometry.mapc2p_center(i,j);
      break;
    case GRAV_CONST: default:
      g[IY] = params.gy;
      break;
  }

  return g;
}

class GravityFunctor {
public:
  Params full_params;
  Geometry geometry;

  GravityFunctor(const Params &full_params) 
    : full_params(full_params), geometry(full_params.device_params) {};
  ~GravityFunctor() = default;

  void applyGravity(Array Q, Array Unew, real_t dt) {
    auto full_params = this->full_params;
    auto params = full_params.device_params;
    auto geometry = this->geometry;

    Kokkos::parallel_for(
      "Gravity", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        Pos g = computeGravity(i, j, params, geometry);

        const real_t r = norm(geometry.mapc2p_center(i,j));
        const real_t rho0 = params.spl_rho(r);
        real_t rhoOld = Q(j, i, IR) * rho0;
        
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


class HeatingFunctor {
  public:
    Params full_params;
    Geometry geometry;
  
    HeatingFunctor(const Params &full_params) 
      : full_params(full_params), geometry(full_params.device_params) {};
    ~HeatingFunctor() = default;
  
    void applyHeating(Array Q, Array Unew, real_t dt) {
      auto full_params = this->full_params;
      auto params = full_params.device_params;
      auto geometry = this->geometry;
      bool no_cooling = params.no_cooling;

      Kokkos::parallel_for(
        "Heating", 
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          Pos pos = geometry.mapc2p_center(i,j);
          const real_t r = norm(pos);
          real_t Q = params.spl_heating.GetValue(r);
          if (no_cooling && Q < 0) Q = 0;
          Unew(j, i, IE) += dt * Q;
        });
    }
  };

class CoriolisFunctor {
  public:
    Params full_params;
  
    CoriolisFunctor(const Params &full_params) 
      : full_params(full_params) {};
    ~CoriolisFunctor() = default;
  
    void applyCoriolis(Array Q, Array Unew, real_t dt) {
      auto full_params = this->full_params;
      auto params = full_params.device_params;
      const real_t omega = params.coriolis_omega; 
  
      Kokkos::parallel_for(
        "Coriolis", 
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          #if 0
            real_t rho = Q(j, i, IR);
            const real_t rhouOld = rho * Q(j, i, IU);
            const real_t rhovOld = rho * Q(j, i, IV); 
            const real_t rhouNew = Unew(j, i, IU);
            const real_t rhovNew = Unew(j, i, IV);
            Unew(j, i, IU) += dt * omega * (rhovOld + rhovNew);
            Unew(j, i, IV) -= dt * omega * (rhouOld + rhouNew);
          #else
            const real_t rhou = Unew(j, i, IU);
            const real_t rhov = Unew(j, i, IV);
            Unew(j, i, IU) += dt * 2 * omega * rhov;
            Unew(j, i, IV) -= dt * 2 * omega * rhou;
          #endif
        });
    }
  };
}

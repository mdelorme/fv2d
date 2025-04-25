#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
Pos computeGravity(int i, int j, const Params &params, const Geometry &geometry) {
  Pos g{0};
  switch (params.gravity) {
    case GRAV_READFILE:
    {
      Pos pos = geometry.mapc2p_center(i,j);
      const real_t r = norm(pos);
      g = params.spl_grav.GetValue(r) / r * pos;
      break;
    }
    case GRAV_CONST: default:
      g[IY] = params.g;
      break;
  }

  return g;
}

class GravityFunctor {
public:
  Params params;
  Geometry geometry;

  GravityFunctor(const Params &params) 
    : params(params), geometry(params) {};
  ~GravityFunctor() = default;

  void applyGravity(Array Q, Array Unew, real_t dt) {
    auto params = this->params;
    auto geo = this->geometry;

    Kokkos::parallel_for(
      "Gravity", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        Pos g = computeGravity(i, j, params, geo);

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

class CoriolisFunctor {
  public:
    Params params;
  
    CoriolisFunctor(const Params &params) 
      : params(params) {};
    ~CoriolisFunctor() = default;
  
    void applyCoriolis(Array Q, Array Unew, real_t dt) {
      auto params = this->params;
      const real_t omega = this->params.coriolis_omega; 
  
      Kokkos::parallel_for(
        "Coriolis", 
        params.range_dom,
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
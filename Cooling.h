#pragma once

#include "SimInfo.h"

namespace fv2d
{

class CoolingFunctor
{
public:
  Params full_params;

  CoolingFunctor(const Params &full_params) : full_params(full_params) {};
  ~CoolingFunctor() = default;

  void applyCooling(Array Q, Array Unew, real_t dt)
  {
    auto params     = full_params.device_params;
    const real_t T0 = params.decay_T0;
    const real_t mu_g = params.decay_mu_g;
    const real_t gamma0 = params.gamma0;

    constexpr real_t mH = 1.6605388628e-24;
    constexpr real_t kB = 1.3806e-16;
    Kokkos::parallel_for(
        "Cooling",
        full_params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          // Infinitely fast cooling functor

          // Taking the values after the hydro update and bringing the temperature to the target
          const real_t rho = Unew(j, i, IR);
          const real_t T_norm = T0 * kB / (mu_g*mH);

          const real_t Ek = 0.5 * (
            Unew(j, i, IU)*Unew(j, i, IU) +
            Unew(j, i, IV)*Unew(j, i, IV)) / rho; 

          if (i==2 && j==2) {
            real_t new_E = Ek + rho * T_norm / (gamma0-1.0);
            real_t old_E = Unew(j, i, IE);
            real_t old_T = (old_E - Ek) * (gamma0-1.0) / rho / kB * (mu_g*mH);
            
            real_t original_Ek = Q(j, i, IR) * 0.5 * (Q(j, i, IU)*Q(j, i, IU) + Q(j, i, IV)*Q(j, i, IV));
            real_t original_E = original_Ek + Q(j, i, IP) / (gamma0-1.0);

            real_t original_T = Q(j, i, IP) / rho / kB * mu_g*mH;
            printf("Cooling; Original E=%.20e; Original Ek=%.20e; Original rho=%e; Original T=%e; Old E=%e; Ek=%e; rho=%e; Old T=%e; New E=%e\n", 
              original_E, original_Ek, Q(j, i, IR), original_T, old_E, Ek, rho, old_T, new_E);
          }
          
          Unew(j, i, IE) = Ek + rho * T_norm / (gamma0-1.0);
        });
  }
};

} // namespace fv2d

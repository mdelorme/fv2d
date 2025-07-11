#pragma once

#include "SimInfo.h"
#include "States.h"

namespace fv2d{

class SourcesFunctor {
public:
    Params full_params;

    SourcesFunctor(const Params &full_params)
        : full_params(full_params) {};
    ~SourcesFunctor() = default;

    void applySources(Array Q, Array Unew, real_t dt, real_t GLM_ch1) const {
        auto params = full_params.device_params;
        const real_t dx = params.dx;
        const real_t dy = params.dy;
        // if (mhd_run && params.riemann_solver == IDEALGLM)
        //     ch_global = ComputeGlobalDivergenceSpeed(Q, full_params);
        Kokkos::parallel_for(
            "Apply sources",
            full_params.range_dom,
            KOKKOS_LAMBDA(const int i, const int j) {
                if (mhd_run && params.div_cleaning == DEDNER) {
                    // Dedner's div-cleaning source term
                    real_t ch = 0.5 * params.CFL * fmin(dx, dy) / dt;
                    real_t cp = std::sqrt(params.cr * ch);
                    real_t parabolic = std::exp(-dt * ch * ch / (cp * cp));
                    Unew(j, i, IPSI) *= parabolic;
                }
                if (params.riemann_solver == IDEALGLM) {
                    const real_t ch_global = params.GLM_scale * GLM_ch1 / dt;
                    const real_t alpha = ch_global/params.cr;
                    // Ideal GLM source term
                    State q = getStateFromArray(Q, i, j);
                    // State qy = swap_component(qx, IY);
                    
                    // TODO: passer à un ordre 2 en espace pour la PLM
                    // Magnetic divergence source terms
                    State SourceMagX {
                        0.0,
                        q[IBX], 
                        q[IBY], 
                        q[IBZ],
                        q[IU]*q[IBX] + q[IV]*q[IBY] + q[IW]*q[IBZ],
                        q[IU], 
                        q[IV], 
                        q[IW],
                        0.0
                    };
                    
                    State SourceMagY {
                        0.0,
                        q[IBY], 
                        q[IBX], 
                        q[IBZ],
                        q[IV]*q[IBY] + q[IU]*q[IBX] + q[IW]*q[IBZ],
                        q[IV], 
                        q[IU], 
                        q[IW],
                        0.0
                    };
                    const real_t dBdx = 0.5 * (Q(j, i+1, IBX) - Q(j, i-1, IBX)) / dx;
                    const real_t dBdy = 0.5 * (Q(j+1, i, IBY) - Q(j-1, i, IBY)) / dy;
                    State SourceMag = dBdx * SourceMagX + dBdy * SourceMagY;
                        
                    // Psi correlated non-conservative term
                    State SourcePsiX {
                        0.0,
                        0.0,
                        0.0,
                        0.0, 
                        q[IU] * q[IPSI], 
                        0.0,
                        0.0,
                        0.0,
                        q[IU]
                    };

                    State SourcePsiY {
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        q[IV] * q[IPSI], 
                        0.0,
                        0.0,
                        0.0,
                        q[IV]
                    };

                    const real_t dPsidx = 0.5 * (Q(j, i+1, IPSI) - Q(j, i-1, IPSI)) / dx;
                    const real_t dPsidy = 0.5 * (Q(j+1, i, IPSI) - Q(j-1, i, IPSI)) / dy;
                    State SourcePsi = dPsidx * SourcePsiX + dPsidy * SourcePsiY;

                    for (int ivar = 0; ivar < Nfields; ++ivar) {
                        Unew(j, i, ivar) += dt * (SourceMag[ivar] + SourcePsi[ivar]);
                    }
                    // Parabolic source term
                    Unew(j, i, IPSI) -= dt * alpha*Q(j, i, IPSI);
                }
            });
    }
};
    
} // namespace fv2d

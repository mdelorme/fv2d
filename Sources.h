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

    void applySources(Array Q, Array Unew, real_t dt) const {
        auto params = full_params.device_params;
        const real_t dx = params.dx;
        const real_t dy = params.dy;
        real_t ch_global = 0.0;
        if (mhd_run && params.riemann_solver == IDEALGLM)
            ch_global = ComputeGlobalDivergenceSpeed(Q, full_params);
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
                    // Ideal GLM source term
                    State q = getStateFromArray(Q, i, j);
                    real_t alpha = ch_global/params.cr;
                    
                    // TODO: passer à un ordre 2 en espace pour la PLM
                    State SMag{}, SPhi{};
                    
                    // Magnetic divergence source term
                    State VMag {0.0, q[IBX], q[IBY], q[IBZ], q[IU]*q[IBX] + q[IV]*q[IBY] + q[IW]*q[IBZ], q[IU], q[IV], q[IW], 0.0};
                    SMag = (0.5 * (Q(j, i+1, IBX) - Q(j, i-1, IBX)) / dx) * VMag + (0.5 * (Q(j+1, i, IBY) - Q(j-1, i, IBY)) / dy) * VMag;
                    
                    // Psi correlated non-conservative term
                    State SPhiX {0.0, 0.0, 0.0, 0.0, q[IU]*q[IPSI], 0.0, 0.0, 0.0, q[IU]};
                    SPhiX *= 0.5 * (Q(j, i+1, IPSI) - Q(j, i-1, IPSI)) / dx;
                    State SPhiY {0.0, 0.0, 0.0, 0.0, q[IV]*q[IPSI], 0.0, 0.0, 0.0, q[IV]};
                    SPhiY *= 0.5 * (Q(j+1, i, IPSI) - Q(j-1, i, IPSI)) / dy;
                    SPhi = SPhiX + SPhiY;
                    
                    // Parabolic source term
                    State SParabolic {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, alpha*q[IPSI]};
                    
                    for (int ivar = 0; ivar < Nfields; ++ivar) {
                        Unew(j, i, ivar) += dt * (SMag[ivar] + SPhi[ivar] + SParabolic[ivar]);
                    }
                }
            });
    }
};
    
} // namespace fv2d

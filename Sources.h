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
                // En fait, les termes sources sont inclus dans le flux
                // NON ! La discrétisation des termes sources est contrainte par la composante en énergie du flux ;
                // On discrétise de manière à ce que f6 = 0, mais ont doit ensuite les ajouter : q^n+1 = qn - dt/dx*(FR-FL) + Source
                    // Ideal GLM source term
                    // State q = getStateFromArray(Q, i, j);
                    // // State qy = swap_component(qx, IY);
                    
                    // // TODO: passer à un ordre 2 en espace pour la PLM
                    // // Magnetic divergence source terms
                    // State SourceMagX {
                    //     0.0,
                    //     q[IBX], 
                    //     q[IBY], 
                    //     q[IBZ],
                    //     q[IU]*q[IBX] + q[IV]*q[IBY] + q[IW]*q[IBZ],
                    //     q[IU], 
                    //     q[IV], 
                    //     q[IW],
                    //     0.0
                    // };
                    
                    // State SourceMagY {
                    //     0.0,
                    //     q[IBY], 
                    //     q[IBX], 
                    //     q[IBZ],
                    //     q[IV]*q[IBY] + q[IU]*q[IBX] + q[IW]*q[IBZ],
                    //     q[IV], 
                    //     q[IU], 
                    //     q[IW],
                    //     0.0
                    // };
                    // const real_t dBdx = 0.5 * (Q(j, i+1, IBX) - Q(j, i-1, IBX)) / dx;
                    // const real_t dBdy = 0.5 * (Q(j+1, i, IBY) - Q(j-1, i, IBY)) / dy;
                    // State SourceMag = dBdx * SourceMagX + dBdy * SourceMagY;
                        
                    // // Psi correlated non-conservative term
                    // State SourcePsiX {
                    //     0.0,
                    //     0.0,
                    //     0.0,
                    //     0.0, 
                    //     q[IU] * q[IPSI], 
                    //     0.0,
                    //     0.0,
                    //     0.0,
                    //     q[IU]
                    // };

                    // State SourcePsiY {
                    //     0.0,
                    //     0.0,
                    //     0.0,
                    //     0.0,
                    //     q[IV] * q[IPSI], 
                    //     0.0,
                    //     0.0,
                    //     0.0,
                    //     q[IV]
                    // };

                    // const real_t dPsidx = 0.5 * (Q(j, i+1, IPSI) - Q(j, i-1, IPSI)) / dx;
                    // // const real_t dPsidy = 0.5 * (Q(j+1, i, IPSI) - Q(j-1, i, IPSI)) / dy;
                    // State SourcePsi = dPsidx * SourcePsiX; // + dPsidy * SourcePsiY;

                    // for (int ivar = 0; ivar < Nfields; ++ivar) {
                    //     Unew(j, i, ivar) += dt * (SourceMag[ivar] + SourcePsi[ivar]);
                    // }
                    // Parabolic source term
                    Unew(j, i, IPSI) -= dt * alpha*Q(j, i, IPSI);
                }
            });
    }
};
    
State IdealGLMSources(State &qL, State &qCL, State &qCR, State &qR, const DeviceParams &params) {
        const real_t dx = params.dx;

        State SourceMagL {
            0.0,
            qL[IBX], 
            qL[IBY], 
            qL[IBZ],
            qL[IU]*qL[IBX] + qL[IV]*qL[IBY] + qL[IW]*qL[IBZ],
            qL[IU], 
            qL[IV], 
            qL[IW],
            0.0
        };
        
        State SourceMagR {
            0.0,
            qR[IBX], 
            qR[IBY], 
            qR[IBZ],
            qR[IU]*qR[IBX] + qR[IV]*qR[IBY] + qR[IW]*qR[IBZ],
            qR[IU], 
            qR[IV], 
            qR[IW],
            0.0
        };
        
        State SourcePsiR {
            0.0,
            0.0,
            0.0,
            0.0, 
            qR[IU] * qR[IPSI], 
            0.0,
            0.0,
            0.0,
            qR[IU]
        };
        
        State SourcePsiL {
            0.0,
            0.0,
            0.0,
            0.0,
            qL[IU] * qL[IPSI], 
            0.0,
            0.0,
            0.0,
            qL[IV]
        };
        const real_t dBdxL   = 0.5 * (qCL[IBX] - qL [IBX]) / dx;
        const real_t dBdxR   = 0.5 * (qR [IBX] - qCR[IBX]) / dx;

        const real_t dPsidxL = 0.5 * (qCL[IPSI] - qL [IPSI]) / dx;
        const real_t dPsidxR = 0.5 * (qR [IPSI] - qCR[IPSI]) / dx;

        State SourceMag = dBdxL*SourceMagL + dBdxR*SourceMagR;
        State SourcePsi = dPsidxL*SourcePsiL + dPsidxR*SourcePsiR;
        return SourceMag + SourcePsi;
    }
} // namespace fv2d

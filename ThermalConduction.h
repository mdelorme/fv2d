#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t computeKappa(int i, int j, const DeviceParams &params) {
  real_t res;
  switch (params.thermal_conductivity_mode) {
    case TCM_B02:
    {
      const real_t y = getPos(params, i, j)[IY];
      const real_t tr = (tanh((y-params.b02_ymid)/params.b02_thickness) + 1.0) * 0.5;
      res = params.kappa * (params.b02_kappa1 * (1.0-tr) + params.b02_kappa2 * tr);
      break;
    }
    // case TCM_READFILE:
    default:
      res = params.kappa;
  }

  return res;
}

class ThermalConductionFunctor {
public:
  Geometry geometry;
  Params full_params;

  ThermalConductionFunctor(const Params &full_params) 
    : full_params(full_params),
      geometry(full_params.device_params) {};
  ~ThermalConductionFunctor() = default;

  void applyThermalConduction(Array Q, Array Unew, real_t dt) {
    auto params = full_params.device_params;
    auto geometry = this->geometry;
    // const real_t kappa = params.kappa;
    const real_t R = params.spl_rho.header.R;

    Kokkos::parallel_for(
      "Thermal conduction", 
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        
        // wb alpha beta : 
        auto getGamma = [&](const real_t r) -> real_t {
          const real_t alpha = params.spl_rho(r);
          const real_t beta = params.spl_prs(r);
          return beta / alpha / R;
        };
        auto getDerivGamma = [&](const real_t r) -> real_t {
          // créé une spline de température ?
          const real_t rho = params.spl_rho(r);
          const real_t prs = params.spl_prs(r);
          const real_t drho = params.spl_rho.GetDerivative(r);
          const real_t dprs = params.spl_prs.GetDerivative(r);
          const real_t dT = (dprs - prs / rho * drho ) / rho / R;
          return dT;
        };
        auto getTemperature = [](Array arr, int i, int j) -> real_t {return arr(j, i, IP) / arr(j, i, IR);};

        auto gradP = computeGradient(Q, getTemperature, j, i,   geometry, params.gradient_type);
        real_t T_P = getTemperature(Q, i, j);
        Pos r_P = geometry.mapc2p_center(i, j);
        
        auto compute_flux = [&](IDir dir, ISide side) {
          int iN = i + (side == ILEFT ? -1 : 1) * (dir == IX);
          int jN = j + (side == ILEFT ? -1 : 1) * (dir == IY);
          
          real_t T_N = getTemperature(Q, iN, jN);
          auto gradN = computeGradient(Q, getTemperature, jN, iN, geometry, params.gradient_type);
          Pos e_PN = geometry.mapc2p_center(iN, jN) - r_P;
          const real_t d_PN = norm(e_PN);
          e_PN = e_PN / d_PN;
          const auto v_Pf = geometry.centerToFace(i, j, dir, side);
          
          const real_t f_P = dot(v_Pf, e_PN) / d_PN;
          const auto gradient_face = (1-f_P)*gradP + f_P*gradN;
          
          const Pos A_f = geometry.getOrientedFaceArea(i, j, dir, side);

          // overrelaxed correction
          const real_t normA_f2 = dot(A_f,A_f);
          const real_t A_d = normA_f2/dot(A_f,e_PN);
          const Pos  A_t = A_f - A_d*e_PN;

          real_t correction = 0;
          { 
            const real_t normA_f = sqrt(normA_f2);
            const Pos f0f = v_Pf - dot(v_Pf, e_PN)*e_PN;

            const Pos f2f = dot(f0f, A_f) * A_f / normA_f2;
            const Pos f0f2 = f0f - f2f;
            if( norm(f0f2) != 0) // si f0, f2, et f ne coincide pas
            {
              const Pos e_T = f0f2 / norm(f0f2);
              const real_t f1f = norm(f0f) * A_d/normA_f;
              correction = dot(gradN - gradP, f1f * e_T) * A_d / d_PN;
            }
          }

          Pos u_r = geometry.faceCenter(i, j, dir, side);
          const real_t r_f = norm(u_r);
          u_r = u_r / r_f;
          const real_t gradT_gamma_f = (A_d * (T_N - T_P) / d_PN + dot(gradient_face, A_t) + correction);
          const real_t gamma_f = getGamma(r_f);
          const real_t dgamma_f = dot(getDerivGamma(r_f) * u_r, A_f);
          const real_t T_gamma_f = (1-f_P)*T_P + f_P*T_N;

          const real_t gradT_f = gamma_f * gradT_gamma_f + T_gamma_f * dgamma_f;
          const real_t kappa_f = params.spl_kappa(r_f);
          return kappa_f * gradT_f;
        };

        // Computing thermal flux
        real_t FL = compute_flux(IX, ILEFT);
        real_t FR = compute_flux(IX, IRIGHT);
        real_t FD = compute_flux(IY, ILEFT);
        real_t FU = compute_flux(IY, IRIGHT);


        auto lambda_bc = [&](real_t &bc_flux, BCTC_Mode bc, IDir dir, ISide side) {
          // const real_t sign = (side == ILEFT) ? -1.0 : 1.0;
          const real_t face_length = norm(geometry.getOrientedFaceArea(i, j, dir, side));
          const real_t rf = norm(geometry.faceCenter(i, j, dir, side));
          const real_t kappa_bc = params.spl_kappa(rf);

          switch (bc) {
            case BCTC_ZERO: 
              bc_flux = 0.0;
              break;
            case BCTC_FIXED_TEMPERATURE: {
              const real_t r = norm(r_P);
              const real_t T_C = T_P * getGamma(r);
              const real_t T_boundary = getGamma(rf);
              const auto dr = rf - r;
              // const real_t r_mid = 0.5 * (r + rf);
              // const real_t kappa_bc = params.spl_kappa(r_mid);
              // printf("%le\t%le\n", T_boundary, T_C);
              bc_flux = face_length * kappa_bc * (T_boundary - T_C) / dr;
              break;
            }
            case BCTC_FIXED_GRADIENT: {
              const real_t dgamma_f = getDerivGamma(rf);
              bc_flux = face_length * kappa_bc * dgamma_f;
              break;
            }
            default: break;
          }
        };
        
//         bool only_once = false; // empeche une double correction des flux sur les ghost diagonales ? 
        // ça a pété pour le fix gradient (température négative)
        // a l'air d'être ok pour le fix température
        if (i==params.ibeg)  {
          lambda_bc(FL, params.bctc_ymin, IX, ILEFT);
//           only_once = true;
        }
        else if (i==params.iend-1) {
          lambda_bc(FR, params.bctc_ymax, IX, IRIGHT);
//           only_once = true;
        }
        if (j==params.jbeg) {
//           if (only_once) 
//             FD = 0;
//           else 
            lambda_bc(FD, params.bctc_ymin, IY, ILEFT);
        }
        else if (j==params.jend-1) {
//           if (only_once) 
//             FU = 0;
//           else 
            lambda_bc(FU, params.bctc_ymax, IY, IRIGHT);
        }

        /** 
         * Boundaries treatment
         * IMPORTANT NOTE :
         * To be accurate, in the case of fixed temperature, since the temperature is taken at the interface
         * the value of kappa should either be averaged between the cell-centered value and the interface
         * or be evaluated at x=0.25dx / x=xmax-0.25dx
         */
        // if (j==params.jbeg && params.bctc_ymin != BCTC_NONE) {
        //   switch (params.bctc_ymin) {
        //     case BCTC_FIXED_TEMPERATURE: FL = kappaL * 2.0 * (TC-params.bctc_ymin_value) / dy; break;
        //     case BCTC_FIXED_GRADIENT:    FL = kappaL * params.bctc_ymin_value; break;
        //     default: break;
        //   }
        // }

        // if (j==params.jend-1 && params.bctc_ymax != BCTC_NONE) {
        //   switch (params.bctc_ymax) {
        //     case BCTC_FIXED_TEMPERATURE: FR = kappaR * 2.0 * (params.bctc_ymax_value-TC) / dy; break;
        //     case BCTC_FIXED_GRADIENT:    FR = kappaR * params.bctc_ymax_value; break;       
        //     default: break;
        //   }
        // }

        real_t V = geometry.cellArea(i,j);

        // And updating using a Godunov-like scheme
        Unew(j, i, IE) += dt * (FL + FR + FU + FD) / V;
      });
  }
};

}

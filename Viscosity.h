#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t computeMu(real_t rf, const DeviceParams &params) {
  switch (params.viscosity_mode) {
    case VSC_PRANDTL_CONSTANT: {
          const real_t Cp = params.spl_rho.header.Cp;
          const real_t prandtl = params.prandtl;
          const real_t kappa = params.spl_kappa(rf);
          return prandtl * kappa / Cp;
    }
    default: return params.mu; break;
  }
}

class ViscosityFunctor {
public:
  Geometry geometry;
  Params full_params;

  ViscosityFunctor(const Params &full_params) 
    : full_params(full_params),
      geometry(full_params.device_params) {};
  ~ViscosityFunctor() = default;

  void applyViscosity(Array Q, Array Unew, real_t dt) {
    auto params = full_params.device_params;
    auto geometry = this->geometry;

    Kokkos::parallel_for(
      "Viscosity",
      full_params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j) {
        auto getVelocity = [](Array arr, int i, int j) -> Pos {return {arr(j, i, IU), arr(j, i, IV)};};

        const auto gradP = computeGradient(Q, getVelocity, j, i, geometry, params.gradient_type);
        const Pos r_P = geometry.mapc2p_center(i, j);
        const Pos velocity_P = getVelocity(Q, i, j);

        auto compute_flux = [&](IDir dir, ISide side) {
          int iN = i + (side == ILEFT ? -1 : 1) * (dir == IX);
          int jN = j + (side == ILEFT ? -1 : 1) * (dir == IY);

          const Pos velocity_N = getVelocity(Q, iN, jN);
          const auto gradN = computeGradient(Q, getVelocity, jN, iN, geometry, params.gradient_type);

        // unit vector from cell P to cell N
          Pos e_PN = geometry.mapc2p_center(iN, jN) - r_P;
          const real_t d_PN = norm(e_PN);
          e_PN = e_PN / d_PN;
          const auto v_Pf = geometry.centerToFace(i, j, dir, side);

          const real_t f_P = dot(v_Pf, e_PN) / d_PN;
          const Kokkos::Array<Pos, 2> gradient_face = {
            (1-f_P)*gradP[IX] + f_P*gradN[IX], 
            (1-f_P)*gradP[IY] + f_P*gradN[IY]
          };
          
          const Pos A_f = geometry.getOrientedFaceArea(i, j, dir, side);
          
          // overrelaxed correction
          const real_t normA_f2 = dot(A_f,A_f);
          const real_t A_d = normA_f2/dot(A_f,e_PN);
          const Pos  A_t = A_f - A_d*e_PN;
          
          Pos f1f_eT {0};
          { 
            const real_t normA_f = sqrt(normA_f2);
            const Pos f0f = v_Pf - dot(v_Pf, e_PN)*e_PN;
            
            const Pos f2f = dot(f0f, A_f) * A_f / normA_f2;
            const Pos f0f2 = f0f - f2f;
            if( norm(f0f2) != 0) // si f0, f2, et f ne coincide pas
            {
              const Pos e_T = f0f2 / norm(f0f2);
              const real_t f1f = norm(f0f) * A_d/normA_f;
              f1f_eT = f1f * e_T;
            }
          }
          /////////////////////////////

          const Kokkos::Array<Pos, 2> grad_recons = {
            (velocity_N[IX] - velocity_P[IX]) * e_PN / d_PN + gradient_face[IX] - dot(gradient_face[IX], e_PN) * e_PN + dot(gradN[IX] - gradP[IX], f1f_eT) * e_PN / d_PN, 
            (velocity_N[IY] - velocity_P[IY]) * e_PN / d_PN + gradient_face[IY] - dot(gradient_face[IY], e_PN) * e_PN + dot(gradN[IY] - gradP[IY], f1f_eT) * e_PN / d_PN
          };
          const Pos velocity_face = (1-f_P)*velocity_P + f_P*velocity_N;
          constexpr real_t four_thirds = 4.0/3.0;
          constexpr real_t two_thirds  = 2.0/3.0;

          const auto& [dudx, dudy] = grad_recons[IX];
          const auto& [dvdx, dvdy] = grad_recons[IY];

          const real_t tau_xx = four_thirds * dudx - two_thirds * dvdy;
          const real_t tau_yy = four_thirds * dvdy - two_thirds * dudx;
          const real_t tau_xy = dudy + dvdx;

          // Building flux
          State flux {0.0, 0.0, 0.0, 0.0};
          const real_t rf = norm(geometry.faceCenter(i, j, dir, side));
          const real_t mu = computeMu(rf, params);

          const real_t F_u = tau_xx * A_f[IX] + tau_xy * A_f[IY];
          const real_t F_v = tau_xy * A_f[IX] + tau_yy * A_f[IY];
          flux[IU] += mu * F_u;
          flux[IV] += mu * F_v;
          flux[IE] += mu * (velocity_face[IX] * F_u + velocity_face[IY] * F_v); 

          return flux;
        };

        State FL = compute_flux(IX, ILEFT);
        State FR = compute_flux(IX, IRIGHT);
        State FD = compute_flux(IY, ILEFT);
        State FU = compute_flux(IY, IRIGHT);

        real_t V = geometry.cellArea(i,j);
        State un_loc = getStateFromArray(Unew, i, j);
        un_loc += dt * (FL + FR + FD + FU) / V;
        setStateInArray(Unew, i, j, un_loc);
      });
  }
};


}
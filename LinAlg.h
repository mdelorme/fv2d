#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t logMean(const real_t xl, const real_t xr, const real_t epsilon = 1e-3){
  const real_t zeta = xl/xr;
  const real_t f = (zeta - 1.0) / (zeta + 1.0);
  const real_t u = f * f;
  real_t F;
  if (u < epsilon){
      F = 1.0 + u/3.0 + u*u/5.0 + u*u*u/7.0;
  }
  else {
      F = Kokkos::log(zeta) / 2.0 / f;
  }
  return (xr + xl) / (2 * F);
}

KOKKOS_INLINE_FUNCTION
State matvecmul(const Matrix& matrix, const State& vector) {
  int rows = Nfields;
  int cols = Nfields;
  State res;
  for (int i=0; i<rows; ++i){
    real_t sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        sum += matrix[i][j] * vector[j];
    }
    res[i] = sum;
    }
    return res;
}


// Matrix matmul(const Matrix &A, const Matrix &B) {
//     Matrix C {};
//     Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nfields, Nfields}),
//                         KOKKOS_LAMBDA(const int i, const int j) {
//                             real_t sum = 0.0;
//                             for (int k = 0; k < Nfields; ++k) {
//                                 sum += A[i][k] * B[k][j];
//                             }
//                             C[i][j] = sum;
//                         });
//     return C;
// }

KOKKOS_INLINE_FUNCTION
real_t dot(const Vect &a, const Vect &b) {
  real_t res = 0.0;
  for (int i = 0; i < 3; ++i) {
      res += a[i] * b[i];
  }
  return res;
}

KOKKOS_INLINE_FUNCTION
real_t norm(const Vect &v) {
  real_t res = 0.0;
  for (int i = 0; i < 3; ++i) {
      res += v[i] * v[i];
  }
  return Kokkos::sqrt(res);
}

} // namespace fv2d

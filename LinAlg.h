#pragma once

#include "SimInfo.h"

namespace fv2d {

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

State matvecmul(const Matrix& matrix, const State& vector) {
  int rows = matrix.extent(0);
  int cols = matrix.extent(1);
  State res;
  for (int i=0; i<rows; ++i){
    real_t sum = 0.0;
    for (int j = 0; j < cols; ++j) {
        sum += matrix(i, j) * vector[j];
    }
    res[i] = sum;
    }
    return res;
}

Matrix matmul(const Matrix &A, const Matrix &B) {
    int rows = A.extent(0);
    int cols = B.extent(1);
    int inner = A.extent(1); // ou B.extent(0)

    // Créer la matrice résultante C
    Matrix C("C", rows, cols);

    // Effectuer la multiplication des matrices
    Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
                        KOKKOS_LAMBDA(int i, int j) {
                            double sum = 0.0;
                            for (int k = 0; k < inner; ++k) {
                                sum += A(i, k) * B(k, j);
                            }
                            C(i, j) = sum;
                        });

    return C;
}

real_t dot(const Vect &a, const Vect &b) {
  real_t res = 0.0;
  for (int i = 0; i < 3; ++i) {
      res += a[i] * b[i];
  }
  return res;
}

real_t norm(const Vect &v) {
  real_t res = 0.0;
  for (int i = 0; i < 3; ++i) {
      res += v[i] * v[i];
  }
  return Kokkos::sqrt(res);
}

} // namespace fv2d

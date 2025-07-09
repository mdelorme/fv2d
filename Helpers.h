#pragma once

namespace fv2d {
using Pos   = Kokkos::Array<real_t, 2>;

KOKKOS_INLINE_FUNCTION
Pos getPos(const DeviceParams& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

KOKKOS_INLINE_FUNCTION
const Pos operator+(const Pos &p, const Pos &q)
{
  return {p[IX] + q[IX],
          p[IY] + q[IY]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator-(const Pos &p, const Pos &q)
{
  return {p[IX] - q[IX],
          p[IY] - q[IY]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator*(real_t f, const Pos &p)
{
  return {f * p[IX],
          f * p[IY]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator*(const Pos &p, real_t f)
{
  return f * p;
}
KOKKOS_INLINE_FUNCTION
const Pos operator/(const Pos &p, real_t f)
{
  return {p[IX] / f,
          p[IY] / f};
}

KOKKOS_INLINE_FUNCTION
real_t dot(const Pos& p, const Pos& q) {
  return p[IX]*q[IX] + p[IY]*q[IY];
}

KOKKOS_INLINE_FUNCTION
real_t norm(const Pos& p) {
  return sqrt(dot(p, p));
}

KOKKOS_INLINE_FUNCTION
real_t dist(const Pos& p, const Pos& q) {
  return norm(p-q);
}

}
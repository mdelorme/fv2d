#pragma once

namespace fv2d {

namespace GeometryMappings {
  /**
   * Radial mapping from Calhoun et al 
   * "Logically Rectangular Grids and Finite Volume Methods for PDEs in Circular and Spherical Domains"
   * SIAM Review, 2008, doi: 10.1137/060664094
   * 
   * @param p position in logical space
   * @param rmax radius of the sphere
   * @return the position in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_radial(const Pos& p, const real_t rmax)
  {
    constexpr real_t epsilon = 1e-10;
    
    real_t d = (fabs(p[IX]) > fabs(p[IY])) ? fabs(p[IX]) : fabs(p[IY]);
    real_t r = sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
    r = (r > epsilon) ? r : epsilon;

    return {
        rmax * d * p[IX] / r,
        rmax * d * p[IY] / r
      };
  }

  /**
   * Deformed cartesian mapping 
   * Based on Colella et al.
   * "High-order, finite-volume methods in mapped coordinates"
   * Journal of Computational Physics, 2011
   * 
   * @param p position in logical space
   * @param deformation_factor deformation of the grid as presented in the paper
   * @return the position in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_deformed_cartesian(const Pos& p, const real_t deformation_factor)
  {
    real_t x = p[IX];
    real_t y = p[IY];

    real_t sins = deformation_factor * sin(2*M_PI * x) * sin(2*M_PI * y);
    
    return {
        (x + sins),
        (y + sins)
      };
  }
  
  /**
   * Ring mapping
   * 
   * @param p position in logical space
   * @param rmin minimum radius of the ring
   * @param rmax maximum radius of the ring
   * @return the position in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_ring(const Pos& p, const real_t rmin, const real_t rmax)
  {
    real_t r = p[IY];
    real_t t = p[IX];

    real_t _cos, _sin;
    _cos = cos(M_PI_2 * t);
    _sin = sin(M_PI_2 * t);

    r = rmin + r * (rmax - rmin)/rmax;
    return {
        r * _cos,
        r * _sin
      };
  }

}

}
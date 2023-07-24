#pragma once

#include "SimInfo.h"

namespace fv2d {

  // mappings
  namespace{
    KOKKOS_INLINE_FUNCTION
    Pos map_radial(const Pos& p)
    {
      constexpr real_t r1 = 1.0;
      constexpr real_t epsilon = 1e-10;
      
      real_t d = (fabs(p[IX]) > fabs(p[IY])) ? fabs(p[IX]) : fabs(p[IY]);
      real_t r = sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
      r = (r > epsilon) ? r : epsilon;

      return {
          r1 * d * p[IX] / r,
          r1 * d * p[IY] / r
        };
    }

    KOKKOS_INLINE_FUNCTION
    Pos map_colella(const Pos& p, const real_t deformation_factor)
    {
      real_t x = p[IX];
      real_t y = p[IY];

      real_t sins = deformation_factor * sin(2*M_PI * x) * sin(2*M_PI * y);
      
      return {
          (x + sins),
          (y + sins)
        };
    }

    KOKKOS_INLINE_FUNCTION
    Pos map_ring(const Pos& p)
    {
      constexpr real_t r0 = 0.5;
      constexpr real_t r1 = 1.0;

      real_t r = p[IY];
      real_t t = p[IX]; // theta

      real_t sin, cos;
      sincos(M_PI_2 * t, &sin, &cos);

      r = r0 + r * (r1 - r0)/r1;
      return {
          r * cos,
          r * sin
        };
    }
  } // anonymous namespace

class Geometry{
public:
  Params params;

  Geometry(const Params &params) 
    : params(params) {};
  ~Geometry() = default;

  KOKKOS_INLINE_FUNCTION
  Pos mapc2p(const Pos& p) const
  {
    switch (params.geometry_type)
    {
      case GEO_RADIAL:     return map_radial(p);
      case GEO_COLELLA:    return map_colella(p, params.geometry_colella_param);
      case GEO_RING:       return map_ring(p);
      case GEO_CARTESIAN:  default: return p;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Pos mapc2p_vertex(int i, int j) const // map vertex of the cell --> bot-left is (i,j)
  {
    return mapc2p(getVertex(i, j));
  }

  KOKKOS_INLINE_FUNCTION
  Pos mapc2p_center(int i, int j) const // map vertex of the cell --> bot-left is (i,j)
  {
    Pos bl = mapc2p_vertex(i  ,j  );
    Pos br = mapc2p_vertex(i+1,j  ); 
    Pos tr = mapc2p_vertex(i+1,j+1); 
    Pos tl = mapc2p_vertex(i  ,j+1);

    return {
      (bl[IX] + br[IX] + tr[IX] + tl[IX]) / 4.0,
      (bl[IY] + br[IY] + tr[IY] + tl[IY]) / 4.0
    };
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellArea(int i, int j) const
  {
    Pos bl = mapc2p_vertex(i  ,j  );
    Pos br = mapc2p_vertex(i+1,j  ); 
    Pos tr = mapc2p_vertex(i+1,j+1); 
    Pos tl = mapc2p_vertex(i  ,j+1);

    return 0.5 * (fabs((tl[IX] - br[IX]) * (tr[IY] - bl[IY]))
                + fabs((tl[IY] - br[IY]) * (tr[IX] - bl[IX])));
  }

  KOKKOS_INLINE_FUNCTION
  Pos interfaceRot(int i, int j, IDir dir, real_t *interface_len) const
  {
    Pos p1 = mapc2p_vertex(i, j);
    Pos p2 = mapc2p_vertex(i + (dir == IY), 
                    j + (dir == IX)); 

    Pos out = {
      p2[IX] - p1[IX],
      p2[IY] - p1[IY],
    };
    *interface_len = sqrt(out[IX] * out[IX] + out[IY] * out[IY]);

      // return {     // riemann_rot
      //    out[IY] / *interface_len,
      //   -out[IX] / *interface_len,
      // };

    if(dir == IX)
      return {
        out[IY] / *interface_len,
        -out[IX] / *interface_len,
      };
    else
      return {
        out[IX] / *interface_len,
        out[IY] / *interface_len,
      };
      
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellLength(int i, int j, IDir dir) const
  {
    const real_t di = (dir == IX) ? 1.0 : 0.5;
    const real_t dj = (dir == IX) ? 0.5 : 1.0;

    Pos p = mapc2p({params.xmin + (i-params.ibeg + ((dir == IX) ? 0.0 : di)) * params.dx,
                    params.ymin + (j-params.jbeg + ((dir == IX) ? dj : 0.0)) * params.dy});

    Pos q = mapc2p({params.xmin + (i-params.ibeg + di) * params.dx,
                    params.ymin + (j-params.jbeg + dj) * params.dy});

    real_t x = (q[IX] - p[IX]);
    real_t y = (q[IY] - p[IY]);

    return sqrt(x*x + y*y);
  }

  KOKKOS_INLINE_FUNCTION
  Pos cellReconsLength(int i, int j, IDir dir) const
  {
    // return {0.5,0.5}; // reconstruct like a cartesian grid
  
    // TODO: verif ---- 'refection' on boundary 
    #if 1
      if(i <= params.ibeg + 1 || i > params.iend ||
        j <= params.jbeg + 1 || j > params.jend )
          return {0.5, 0.5};
    #elif 0
      if(i <= params.ibeg + 1 || i > params.iend ||
        j <= params.jbeg + 1 || j > params.jend )
          return {0.5, 0.5};
    #endif

    Pos l, c, r;
    if(dir == IX)
    {
      l = mapc2p({params.xmin + (i-params.ibeg - 1.0) * params.dx,
                  params.ymin + (j-params.jbeg + 0.5) * params.dy});
      c = mapc2p({params.xmin + (i-params.ibeg + 0.0) * params.dx,
                  params.ymin + (j-params.jbeg + 0.5) * params.dy});
      r = mapc2p({params.xmin + (i-params.ibeg + 1.0) * params.dx,
                  params.ymin + (j-params.jbeg + 0.5) * params.dy});
    }
    else
    {
      l = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
                  params.ymin + (j-params.jbeg - 1.0) * params.dy});
      c = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
                  params.ymin + (j-params.jbeg + 0.0) * params.dy});
      r = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
                  params.ymin + (j-params.jbeg + 1.0) * params.dy});
    }
    
    real_t dL, dR;
    {
      real_t x, y;
      // left
      x = c[IX] - l[IX];
      y = c[IY] - l[IY];
      dL = sqrt(x*x + y*y);
      // right
      x = r[IX] - c[IX];
      y = r[IY] - c[IY];
      dR = sqrt(x*x + y*y);
    }

    return {dL, dR};
    // return {dL / (dL + dR),
    //         dR / (dL + dR)};
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellReconsLengthSlope(int i, int j, IDir dir) const
  {
    Pos p = cellReconsLength(i, j, dir);
    return p[0] + p[1];
  }

  private:
    // return vertex at the bot-left of the cell (i,j)
    KOKKOS_INLINE_FUNCTION
    Pos getVertex(int i, int j) const{ 
      return {params.xmin + (i-params.ibeg) * params.dx,
              params.ymin + (j-params.jbeg) * params.dy};
    }
    Pos getCenter(int i, int j) const{ 
      return {params.xmin + (i-params.ibeg+0.5) * params.dx,
              params.ymin + (j-params.jbeg+0.5) * params.dy};
    }
};
}

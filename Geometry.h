#pragma once

#include "SimInfo.h"

namespace fv2d {

  // mappings
  namespace{
    KOKKOS_INLINE_FUNCTION
    real_t norm(const Pos& p) {
      return sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
    }

    KOKKOS_INLINE_FUNCTION
    Pos map_radial(const Pos& p, const real_t R)
    {
      constexpr real_t epsilon = 1e-10;
      
      real_t d = (fabs(p[IX]) > fabs(p[IY])) ? fabs(p[IX]) : fabs(p[IY]);
      real_t r = sqrt(p[IX]*p[IX] + p[IY]*p[IY]);
      r = (r > epsilon) ? r : epsilon;

      return {
          R * d * p[IX] / r,
          R * d * p[IY] / r
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

      real_t _cos, _sin;
      // sincos(M_PI_2 * t, &_sin, &_cos);
      _cos = cos(M_PI_2 * t);
      _sin = sin(M_PI_2 * t);

      r = r0 + r * (r1 - r0)/r1;
      return {
          r * _cos,
          r * _sin
        };
    }
    
    KOKKOS_INLINE_FUNCTION
    Pos map_test(const Pos& p, const real_t rot)
    {
      const real_t t = rot * M_PI;

      real_t x = p[IX];
      real_t y = p[IY];

      real_t _cos, _sin;
      _cos = cos(t);
      _sin = sin(t);

      return {
          x * _cos - y * _sin,
          x * _sin + y * _cos
        };
    }
  } // anonymous namespace

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
      case GEO_RADIAL:     return map_radial(p, params.radial_radius);
      case GEO_COLELLA:    return map_colella(p, params.geometry_colella_param);
      case GEO_RING:       return map_ring(p);
      case GEO_TEST:       return map_test(p, params.geometry_colella_param);
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

    switch(params.mapc2p_type) {
      case MAP_MAPPED:
        return mapc2p(getCenter(i,j));
      case MAP_CENTER:
        return 0.25 * (bl + br + tr + tl);
      case MAP_CENTROID: default:
        const real_t vol = (bl[IX]*br[IY] - bl[IX]*tl[IY] - br[IX]*bl[IY] + br[IX]*tr[IY] - tr[IX]*br[IY] + tr[IX]*tl[IY] + tl[IX]*bl[IY] - tl[IX]*tr[IY]);
        real_t cx = ((bl[IX] + br[IX])*(bl[IX]*br[IY] - br[IX]*bl[IY]) - (bl[IX] + tl[IX])*(bl[IX]*tl[IY] - tl[IX]*bl[IY]) + (br[IX] + tr[IX])*(br[IX]*tr[IY] - tr[IX]*br[IY]) + (tr[IX] + tl[IX])*(tr[IX]*tl[IY] - tl[IX]*tr[IY])) / (3*vol);
        real_t cy = ((bl[IY] + br[IY])*(bl[IX]*br[IY] - br[IX]*bl[IY]) - (bl[IY] + tl[IY])*(bl[IX]*tl[IY] - tl[IX]*bl[IY]) + (br[IY] + tr[IY])*(br[IX]*tr[IY] - tr[IX]*br[IY]) + (tr[IY] + tl[IY])*(tr[IX]*tl[IY] - tl[IX]*tr[IY])) / (3*vol);
        return {cx, cy};
    }
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellArea(int i, int j) const
  {
    Pos bl = mapc2p_vertex(i  ,j  );
    Pos br = mapc2p_vertex(i+1,j  ); 
    Pos tr = mapc2p_vertex(i+1,j+1); 
    Pos tl = mapc2p_vertex(i  ,j+1);

    // return 0.5 * (fabs((tl[IX] - br[IX]) * (tr[IY] - bl[IY]))
    //             + fabs((tl[IY] - br[IY]) * (tr[IX] - bl[IX])));
    return 0.5 * fabs((tl[IX] - br[IX]) * (tr[IY] - bl[IY]) -
                      (tl[IY] - br[IY]) * (tr[IX] - bl[IX]));
  }

/*
  KOKKOS_INLINE_FUNCTION
  Pos interfaceRot(int i, int j, IDir dir, real_t *interface_len) const
  {
    Pos p, q;
    if(dir == IX)
    {
      p = mapc2p_vertex(i, j);
      q = mapc2p_vertex(i, j+1); 
    }
    else
    {
      p = mapc2p_vertex(i+1, j);
      q = mapc2p_vertex(i, j); 
    }

    Pos tanj = q-p;
    *interface_len = sqrt(tanj[IX] * tanj[IX] + tanj[IY] * tanj[IY]);


    // normal at the interface
    return {
        tanj[IY] / *interface_len,
       -tanj[IX] / *interface_len,
    };

    // if(dir == IX)
    //   return {
    //      tanj[IY] / *interface_len,
    //     -tanj[IX] / *interface_len,
    //   };
    // else
    //   return {
    //     tanj[IX] / *interface_len,
    //     tanj[IY] / *interface_len,
    //   };
      
  }
*/

  KOKKOS_INLINE_FUNCTION
  Pos getRotationMatrix(int i, int j, IDir dir, ISide side, real_t &interface_len) const
  {
    Pos tj; // tangential vector to the face
    switch(dir)
    {
      case IX:
        if(side == ILEFT)
          tj = mapc2p_vertex(i,   j+1) - mapc2p_vertex(i,   j);
        else
          tj = mapc2p_vertex(i+1, j+1) - mapc2p_vertex(i+1, j);
        break;
      default:
        if(side == ILEFT)
          tj = mapc2p_vertex(i, j)     - mapc2p_vertex(i+1, j);
        else
          tj = mapc2p_vertex(i, j+1)   - mapc2p_vertex(i+1, j+1);
        break;
    }
    interface_len = norm(tj);
    return {
       tj[IY] / interface_len,
      -tj[IX] / interface_len
    };
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellLength(int i, int j, IDir dir) const
  {
    // const real_t di = (dir == IX) ? 1.0 : 0.5;
    // const real_t dj = (dir == IX) ? 0.5 : 1.0;

    // Pos p = mapc2p({params.xmin + (i-params.ibeg + ((dir == IX) ? 0.0 : di)) * params.dx,
    //                 params.ymin + (j-params.jbeg + ((dir == IX) ? dj : 0.0)) * params.dy});

    // Pos q = mapc2p({params.xmin + (i-params.ibeg + di) * params.dx,
    //                 params.ymin + (j-params.jbeg + dj) * params.dy});
    
  #if 1

    Pos bl = mapc2p_vertex(i  ,j  );
    Pos br = mapc2p_vertex(i+1,j  ); 
    Pos tr = mapc2p_vertex(i+1,j+1); 
    Pos tl = mapc2p_vertex(i  ,j+1);
  
    Pos l, r;
    if(dir == IX)
    {
      l = 0.5 * (bl + tl);
      r = 0.5 * (br + tr);
    }
    else
    {
      l = 0.5 * (bl + br);
      r = 0.5 * (tl + tr);
    }
    return norm(l-r);
  
  #else

    Pos l, c, r;
    if(dir == IX)
    {
      l = 0.5 * (bl + tl);
      c = 0.25 * (bl + br + tr + tl);
      r = 0.5 * (br + tr);
    }
    else
    {
      l = 0.5 * (bl + br);
      c = 0.25 * (bl + br + tr + tl);
      r = 0.5 * (tl + tr);
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

    return dL + dR;
    #endif
  }

  KOKKOS_INLINE_FUNCTION
  Pos cellReconsLength(int i, int j, IDir dir) const
  {
    // return {0.5,0.5}; // reconstruct like a cartesian grid
  
    // TODO: verif ---- 'refection' on boundary 
    #if 0
      if(i <= params.ibeg + 1 || i > params.iend ||
        j <= params.jbeg + 1 || j > params.jend )
          return {0.5, 0.5};
    #elif 0
      if(i <= params.ibeg + 1 || i > params.iend ||
        j <= params.jbeg + 1 || j > params.jend )
          return {0.5, 0.5};
    #endif

    Pos l, c, r;
    // if(dir == IX)
    // {
    //   l = mapc2p({params.xmin + (i-params.ibeg - 1.0) * params.dx,
    //               params.ymin + (j-params.jbeg + 0.5) * params.dy});
    //   c = mapc2p({params.xmin + (i-params.ibeg + 0.0) * params.dx,
    //               params.ymin + (j-params.jbeg + 0.5) * params.dy});
    //   r = mapc2p({params.xmin + (i-params.ibeg + 1.0) * params.dx,
    //               params.ymin + (j-params.jbeg + 0.5) * params.dy});
    // }
    // else
    // {
    //   l = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
    //               params.ymin + (j-params.jbeg - 1.0) * params.dy});
    //   c = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
    //               params.ymin + (j-params.jbeg + 0.0) * params.dy});
    //   r = mapc2p({params.xmin + (i-params.ibeg + 0.5) * params.dx,
    //               params.ymin + (j-params.jbeg + 1.0) * params.dy});
    // }

    if(dir == IX)
    {
      l = mapc2p_center(i-1, j);
      c = 0.5 * (mapc2p_vertex(i,j) + mapc2p_vertex(i,j+1));
      r = mapc2p_center(i, j);
    }
    else
    {
      l = mapc2p_center(i, j-1);
      c = 0.5 * (mapc2p_vertex(i,j) + mapc2p_vertex(i+1,j));
      r = mapc2p_center(i, j);
    }

    real_t dL = norm(c-l);
    real_t dR = norm(r-c);

    return {dL, dR};
    // return {dL / (dL + dR),
    //         dR / (dL + dR)};
  }

  KOKKOS_INLINE_FUNCTION
  real_t cellReconsLengthSlope(int i, int j, IDir dir) const
  {
    Pos p = mapc2p_center(i, j);
    Pos q;
    if (dir == IX) q = mapc2p_center(i-1, j);
    else           q = mapc2p_center(i, j-1);
    return norm(p-q);
    
    // Pos p = cellReconsLength(i, j, dir);
    // return p[0] + p[1];
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

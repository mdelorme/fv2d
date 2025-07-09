#pragma once

#include "SimInfo.h"
#include "GeometryMappings.h"

namespace fv2d {

class Geometry {
public:
  /**
   * Face information in physical space
   */
  struct FaceInfo {
    Pos normal;               // (non oriented) Normal at the face 
    real_t interface_length;  // Length of the face
  };

  DeviceParams params;

  Geometry(const DeviceParams &params) 
    : params(params) {};
  ~Geometry() = default;

  /**
   * Mapping function from cartian/logical to physical geometry
   * 
   * @param p Position of the point in logical space
   * @return Position of the point in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_to_physical(const Pos& p) const
  {
    switch (params.geometry_type)
    {
      case GEO_RADIAL:             return GeometryMappings::map_radial(p, params.rmax);
      case GEO_DEFORMED_CARTESIAN: return GeometryMappings::map_deformed_cartesian(p, params.geometry_deformation);
      case GEO_RING:               return GeometryMappings::map_ring(p, params.rmin, params.rmax);
      case GEO_CARTESIAN: default: return p;
    }
  }

  /**
   * Mapping function from cartesian to physical for a vertex
   * 
   * Cell index (i, j) corresponds to the bottom left of the cell
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @return Position of the vertex in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_to_physical_vertex(int i, int j) const
  {
    const Pos logical_pos {params.xmin + (i-params.ibeg) * params.dx,
                           params.ymin + (j-params.jbeg) * params.dy};

    return map_to_physical(logical_pos);
  }

  /**
   * Mapping function from cartesian to physical for the center of a cell
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @return Position of the center of the cell in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos map_to_physical_center(int i, int j) const
  {
    const Pos bl = map_to_physical_vertex(i  ,j  );
    const Pos br = map_to_physical_vertex(i+1,j  ); 
    const Pos tr = map_to_physical_vertex(i+1,j+1); 
    const Pos tl = map_to_physical_vertex(i  ,j+1);
    return 0.25 * (bl + br + tr + tl);
  }

  /**
   * Computes the area of a physical cell according to geometry
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @return Area of the cell in the physical space
   */
  KOKKOS_INLINE_FUNCTION
  real_t cellArea(int i, int j) const
  {
    const Pos bl = map_to_physical_vertex(i  ,j  );
    const Pos br = map_to_physical_vertex(i+1,j  ); 
    const Pos tr = map_to_physical_vertex(i+1,j+1); 
    const Pos tl = map_to_physical_vertex(i  ,j+1);

    return 0.5 * fabs((tl[IX] - br[IX]) * (tr[IY] - bl[IY]) -
                      (tl[IY] - br[IY]) * (tr[IX] - bl[IX]));
  }

  /**
   * Returns the oriented info of a face in physical space
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interface
   * @param side side of the interface 
   * @return the outward pointing normal of an interface in physical space
   */
  KOKKOS_INLINE_FUNCTION
  Pos getOrientedFaceArea(int i, int j, IDir dir, ISide side) const
  {
    Pos tj; // tangential vector to the face
    const int iix[2][2][2] {{{i, i+1}, {i+1, i}},
                            {{i, i+1}, {i, i+1}}};
    const int iiy[2][2][2] {{{j, j+1}, {j, j+1}},
                            {{j+1, j}, {j, j+1}}};
    
    tj =   map_to_physical_vertex(iix[0][dir][side], iiy[0][dir][side]) 
         - map_to_physical_vertex(iix[1][dir][side], iiy[1][dir][side]);

    return {tj[IY], -tj[IX]}; 
  }

  /**
   * Returns the info of a face in physical space
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interface
   * @param side side of the interface 
   * @return a FaceInfo object that stores the normal and the interface length in physical space
   */
  KOKKOS_INLINE_FUNCTION
  FaceInfo getFaceInfo(int i, int j, IDir dir, ISide side) const
  {
    const int sign = (side == ILEFT) ? -1 : 1;
    const auto normal = getOrientedFaceArea(i, j, dir, side);
    const real_t interface_len = norm(normal);
    return {
      sign * normal / interface_len,
      interface_len
    };
  }

  /**
   * Calculates the distance between two opposite faces in a cell in physical space
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interace
   * @return the distance between the centers of two opposite faces
   */
  KOKKOS_INLINE_FUNCTION
  real_t getCellLength(int i, int j, IDir dir) const
  {
    const Pos bl = map_to_physical_vertex(i  ,j  );
    const Pos br = map_to_physical_vertex(i+1,j  ); 
    const Pos tr = map_to_physical_vertex(i+1,j+1); 
    const Pos tl = map_to_physical_vertex(i  ,j+1);
  
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
  }

  /**
   * Calculates the reconstruction lengths each side of the "left" interface along a direction
   * 
   * In practice, if dir == IX, returns the distance between cell center i-1 and interface i-1/2 
   * and cell center i and interface i-1/2
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interace
   * @return the left and right reconstruction lengths
   */
  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<real_t, real_t> getCellReconsLenghts(int i, int j, IDir dir) const
  {
    Pos l, c;

    if(dir == IX)
    {
      l = map_to_physical_center(i-1, j);
      c = 0.5 * (map_to_physical_vertex(i,j) + map_to_physical_vertex(i,j+1));
    }
    else
    {
      l = map_to_physical_center(i, j-1);
      c = 0.5 * (map_to_physical_vertex(i,j) + map_to_physical_vertex(i+1,j));
    }

    const Pos r = map_to_physical_center(i, j);

    const real_t dL = norm(c-l);
    const real_t dR = norm(r-c);

    return {dL, dR};
  }

  /**
   * Calculates the distance between the centers of two cells
   * 
   * The calculation is always made with respect to the cell with index -1 along dir
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interace
   * @return the distance between the center of two cells
   */
  KOKKOS_INLINE_FUNCTION
  real_t getCellReconsLengthSlope(int i, int j, IDir dir) const
  {
    Pos p = map_to_physical_center(i, j);
    Pos q = (dir == IX) ? map_to_physical_center(i-1, j) : map_to_physical_center(i, j-1);
    return norm(p-q);
  }

  /**
   * Calculates the center point of a face
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interface
   * @param side side of the interface 
   * @return the position of the center of the face
   */
  KOKKOS_INLINE_FUNCTION
  Pos getFaceCenter(int i, int j, IDir dir, ISide side) const
  {
    const Pos bl = map_to_physical_center(i  ,j  );
    const Pos br = map_to_physical_center(i+1,j  ); 
    const Pos tr = map_to_physical_center(i+1,j+1); 
    const Pos tl = map_to_physical_center(i  ,j+1);

    if      (dir == IX && side == ILEFT)
      return 0.5 * (bl + tl);
    else if (dir == IX && side == IRIGHT)
      return 0.5 * (br + tr);
    else if (dir == IY && side == ILEFT)
      return 0.5 * (bl + br);
    else if (dir == IY && side == IRIGHT)
      return 0.5 * (tl + tr);
    else {
      Kokkos::abort("unknown faceCenter");
    }
  }
  
  /** 
   * Calculates the vector from a center to a face 
   * 
   * @param i cell index along X-axis
   * @param j cell index along Y-axis
   * @param dir axis of the interface
   * @param side side of the interface 
   * @return the vector going from the center of a cell to the center of a face
   **/
  KOKKOS_INLINE_FUNCTION
  Pos getCenterToFace(int i, int j, IDir dir, ISide side) const
  {
    return getFaceCenter(i,j,dir,side) - map_to_physical_center(i, j);
  }
};

} // namespace fv2d
#pragma once

#include "SimInfo.h"
#include "Geometry.h"

namespace fv2d {

template<bool     wide_stencil,
         typename State,
         typename FunctionAccessor,
         int ndim=2>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<State, ndim> LeastSquareGradient(const Array&              Q,
                                               const FunctionAccessor&   getState,
                                               const int                 i,
                                               const int                 j,
                                               const Geometry&           geometry )
{

  constexpr int nb_neighbors = wide_stencil ? 8 : 4;
  Kokkos::Array<State, nb_neighbors> stencil;
  Kokkos::Array<Pos,   nb_neighbors> distances;

  // fill stencil & distance matrix
  {
    Pos main_centroid = geometry.mapc2p_center(i,j);
    State main_state = getState(Q, i, j);
    int id = 0;

    if constexpr(wide_stencil)
    {
      for (int ii : {i-1, i, i+1}) {
        for (int jj : {j-1, j, j+1}) {
          if(ii==i && jj==j) 
            continue;
          
          stencil[id] = getState(Q, ii, jj) - main_state;
          distances[id] = geometry.mapc2p_center(ii,jj) - main_centroid;
          id++;
        }
      }
    }
    else // small stencil
    {
      for (IDir dir : {IX, IY}){
        for (int side : {-1, +1}){
          int ii = i + (dir == IX ? side : 0);
          int jj = j + (dir == IY ? side : 0);

          stencil[id] = getState(Q, ii, jj) - main_state;
          distances[id] = geometry.mapc2p_center(ii,jj) - main_centroid;
          id++;
        }
      }
    }
  }

  // construct DD matrix : D^T*D
  real_t DD_Mat[ndim][ndim];
  for (int i=0; i<ndim; ++i)
    for (int j=0; j<ndim; ++j){
      real_t dd = 0;
      for (int k=0; k<nb_neighbors; k++)
      {
        dd += distances[k][i] * distances[k][j];
      }
      DD_Mat[i][j] = dd;
    }
    
  // inverse of (D^T * D) -> invDD_Mat
  Kokkos::Array<real_t, ndim*ndim> invDD_Mat;
  real_t det_DD;
  {
    const real_t *a = DD_Mat[0];
    const real_t *b = DD_Mat[1];
    if constexpr(ndim == 2)
    {
      det_DD = a[0]*b[1] - a[1]*b[0];
      invDD_Mat = { b[1], -a[1],
                   -b[0],  a[0]};
    }
    else /* ndim == 3 */
    {
      printf("unreachable");
      exit(0);
    }
  }

  // compute G matrix : G = (D^T D)^-1 * D^T
  real_t G_Mat[ndim][nb_neighbors];
  for (int i=0; i<ndim; ++i)
    for (int k=0; k<nb_neighbors; k++){
      real_t g = 0; // current elem of G
      for (int j=0; j<ndim; ++j)
      {
        g += invDD_Mat[i*ndim + j] * distances[k][j];
      }
      G_Mat[i][k] = g;
    }
  
  // compute gradient
  using Gradient = Kokkos::Array<State, ndim>;
  Gradient gradient = {0};
  for (int i=0; i<ndim; ++i){
    State gi{0};
    for (int k=0; k<nb_neighbors; k++){
      gi += G_Mat[i][k] * stencil[k];
    }
    gradient[i] = gi / det_DD;
  }

  return gradient;
}

template<typename State,
         typename FunctionAccessor,
         int ndim=2>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<State, ndim> LeastSquareNodeGradient(const Array&              Q,
                                                   const FunctionAccessor&   getState,
                                                   const int                 i,
                                                   const int                 j,
                                                   const Geometry&           geometry )
{
  constexpr int nb_neighbors = 4;
  Kokkos::Array<State, nb_neighbors> node_values;
  Kokkos::Array<Pos,   nb_neighbors> distances;

  // fill stencil & distance matrix
  {
    // fill stencil & centroids
    Kokkos::Array<State, 3*3> stencil;
    Kokkos::Array<Pos,   3*3> centroids;
    Pos main_center = geometry.mapc2p_center(i,j);
    State main_state = getState(Q, i, j);
    
    for (int k=0; k<=2; k++) {
      for (int l=0; l<=2; l++) {
        int ii = i + k - 1;
        int jj = j + l - 1;
        stencil[3*k + l] = getState(Q, ii, jj);
        centroids[3*k + l] = geometry.mapc2p_center(ii,jj);
      }
    }
  
    for (int k : {0, 1}) {
      for (int l : {0, 1}) {
        Pos node_pos = geometry.getNode(i+k,j+l);
        real_t sum_inv_distance = 0;
        State node_value = {0};
  
        for (int kk : {k, k+1}) {
          for (int ll : {l, l+1}) {
            real_t inv_distance =  1 / dist(centroids[3*kk+ll], node_pos);
            node_value += stencil[3*kk+ll] * inv_distance;
            sum_inv_distance += inv_distance;
          }
        }
        distances[2*k+l] = node_pos - main_center;
        node_values[2*k+l] = ((node_value-main_state) / sum_inv_distance);
      }
    }
  }

  // construct DD matrix : D^T*D
  real_t DD_Mat[ndim][ndim];
  for (int i=0; i<ndim; ++i)
    for (int j=0; j<ndim; ++j){
      real_t dd = 0;
      for (int k=0; k<nb_neighbors; k++)
      {
        dd += distances[k][i] * distances[k][j];
      }
      DD_Mat[i][j] = dd;
    }
    
  // inverse of (D^T * D) -> invDD_Mat
  Kokkos::Array<real_t, ndim*ndim> invDD_Mat;
  real_t det_DD;
  {
    const real_t *a = DD_Mat[0];
    const real_t *b = DD_Mat[1];
    if constexpr(ndim == 2)
    {
      det_DD = a[0]*b[1] - a[1]*b[0];
      invDD_Mat = { b[1], -a[1],
                  -b[0],  a[0]};
    }
    else /* ndim == 3 */
    {
      printf("unreachable");
      exit(0);
    }
  }

  // compute G matrix : G = (D^T D)^-1 * D^T
  real_t G_Mat[ndim][nb_neighbors];
  for (int i=0; i<ndim; ++i)
    for (int k=0; k<nb_neighbors; k++){
      real_t g = 0; // current elem of G
      for (int j=0; j<ndim; ++j)
      {
        g += invDD_Mat[i*ndim + j] * distances[k][j];
      }
      G_Mat[i][k] = g;
    }
  
  // compute gradient
  using Gradient = Kokkos::Array<State, ndim>;
  Gradient gradient = {0};
  for (int i=0; i<ndim; ++i){
    State gi{0};
    for (int k=0; k<nb_neighbors; k++){
      gi += G_Mat[i][k] * node_values[k];
    }
    gradient[i] = gi / det_DD;
  }

  return gradient;
}


template<typename State,
         typename FunctionAccessor,
         int ndim=2>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<State, ndim> CellBasedGradient(const Array&              Q,
                                             const FunctionAccessor&   getState,
                                             const int                 i,
                                             const int                 j,
                                             const Geometry&           geometry )
{
  using Gradient = Kokkos::Array<State, ndim>;

  State main_state = getState(Q, i, j);
  Pos main_centroid = geometry.mapc2p_center(i,j);
  real_t area = geometry.cellArea(i, j);

  // gradient
  Gradient gradient = {0};
  for (IDir dir : {IX, IY}){
    for (ISide side : {ILEFT, IRIGHT}) { 
      const int sign = (side == ILEFT) ? -1 : 1;
      int ii = i + (dir == IX ? sign : 0);
      int jj = j + (dir == IY ? sign : 0);

      State neighbor_state = getState(Q, ii, jj);

      real_t f;
      {
        auto face_center = geometry.faceCenter(i,j, dir, side);
        real_t f1 = dist(face_center, geometry.mapc2p_center(ii,jj));
        real_t f2 = dist(face_center, main_centroid);
        f = f1 / (f1 + f2);
      }
      State facevalue = (1-f) * neighbor_state + f * main_state;

      // real_t interface_len;
      // Pos n = geometry.getRotationMatrix(i,j,dir,side,interface_len);
      // gradient[IX] += sign * facevalue * interface_len * n[IX];
      // gradient[IY] += sign * facevalue * interface_len * n[IY];

      Pos n = geometry.getOrientedFaceArea(i,j,dir,side);
      gradient[IX] += facevalue * n[IX];
      gradient[IY] += facevalue * n[IY];
    }
  } 
  gradient[IX] = gradient[IX] / area;
  gradient[IY] = gradient[IY] / area;

  return gradient;
}


template<typename State,
         typename FunctionAccessor,
         int ndim=2>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<State, ndim> NodeBasedGradient(const Array&              Q,
                                             const FunctionAccessor&   getState,
                                             const int                 i,
                                             const int                 j,
                                             const Geometry&           geometry )
{
  // fill stencil & centroids
  Kokkos::Array<State, 3*3> stencil;
  Kokkos::Array<Pos,   3*3> centroids;
  
  for (int k=0; k<=2; k++) {
    for (int l=0; l<=2; l++) {
      int ii = i + k - 1;
      int jj = j + l - 1;
      stencil[3*k + l] = getState(Q, ii, jj);
      centroids[3*k + l] = geometry.mapc2p_center(ii,jj);
    }
  }

  // compute nodebased gradient
  Kokkos::Array<State, 2*2> node_values;

  for (int k : {0, 1}) {
    for (int l : {0, 1}) {
      Pos node_pos = geometry.getNode(i+k,j+l);
      real_t sum_inv_distance = 0;
      State node_value = {0};

      for (int kk : {k, k+1}) {
        for (int ll : {l, l+1}) {
          real_t inv_distance =  1 / dist(centroids[3*kk+ll], node_pos);
          node_value += stencil[3*kk+ll] * inv_distance;
          sum_inv_distance += inv_distance;
        }
      }
      node_values[2*k+l] = node_value / sum_inv_distance;
    }
  }

  using Gradient = Kokkos::Array<State, ndim>;
  Gradient gradient = {0};
  for (IDir dir : {IX, IY}) {
    for(ISide side : {ILEFT, IRIGHT}) {
      const int s = (side == ILEFT) ?  0 : 1;
      const int id1 = (dir == IX) ? 2*s   :   s;
      const int id2 = (dir == IX) ? 2*s+1 : 2+s;

      State facevalue = (node_values[id1] + node_values[id2]) / 2.;
      Pos n = geometry.getOrientedFaceArea(i,j,dir,side);
      gradient[IX] += facevalue * n[IX];
      gradient[IY] += facevalue * n[IY];
    }
  }
  
  const real_t area = geometry.cellArea(i, j);
  gradient[IX] = gradient[IX] / area;
  gradient[IY] = gradient[IY] / area;
  
  return gradient;
}

template<typename FunctionAccessor, int ndim=2> KOKKOS_INLINE_FUNCTION
auto computeGradient(const Array& Q, const FunctionAccessor& getState, const int i, const int j, const Geometry& geometry, const GradientType gradient_type)
{
  using State = decltype(getState(Q, 0, 0));
  switch(gradient_type) {
    case LEAST_SQUARE_WIDE: return LeastSquareGradient<true,  State>(Q, getState, i, j, geometry);
    case LEAST_SQUARE:      return LeastSquareGradient<false, State>(Q, getState, i, j, geometry);
    // case LEAST_SQUARE_NODE: return LeastSquareNodeGradient<State>(Q, getState, i, j, geometry);
    case GREEN_GAUSS:       return CellBasedGradient<State>(Q, getState, i, j, geometry);
    case GREEN_GAUSS_WIDE:  return NodeBasedGradient<State>(Q, getState, i, j, geometry);
    default: Kokkos::abort("unknown gradient type");
  }
}

}


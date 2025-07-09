#pragma once

#include "SimInfo.h"
#include "Geometry.h"

namespace fv2d {

template<bool     wide_stencil,
         typename State,
         typename FunctionAccessor>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<State, 2> LeastSquareGradient(const Array&              Q,
                                            const FunctionAccessor&   getState,
                                            const int                 i,
                                            const int                 j,
                                            const Geometry&           geometry )
{

  constexpr int nb_neighbors = wide_stencil ? 8 : 4;
  Kokkos::Array<State, nb_neighbors> stencil;
  Kokkos::Array<Pos,   nb_neighbors> distances;

  constexpr int ndim = 2;

  // fill stencil & distance matrix
  {
    Pos main_centroid = geometry.map_to_physical_center(i,j);
    State main_state = getState(Q, i, j);
    int id = 0;

    if constexpr(wide_stencil)
    {
      for (int ii : {i-1, i, i+1}) {
        for (int jj : {j-1, j, j+1}) {
          if(ii==i && jj==j) 
            continue;
          
          stencil[id] = getState(Q, ii, jj) - main_state;
          distances[id] = geometry.map_to_physical_center(ii,jj) - main_centroid;
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
          distances[id] = geometry.map_to_physical_center(ii,jj) - main_centroid;
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


/**
 * Computes the gradient of an array
 * 
 * @tparam FunctionAccessor how to get the state in Q
 * @param Q array of primitive variables 
 * @param getState function returning a "State" from Q and indices i, j
 * @param i cell index along X-axis
 * @param j cell index along Y-axis
 * @param geometry geometry object for the grid
 * @return 2 states in an array, dState/dx, dState/dy
 */
template <typename FunctionAccessor>
KOKKOS_INLINE_FUNCTION
auto computeGradient(const Array& Q, const FunctionAccessor& getState, const int i, const int j, const Geometry& geometry)
{
  using State = decltype(getState(Q, 0, 0));
  return LeastSquareGradient<true, State>(Q, getState, i, j, geometry);
}

} // namespace fv2d
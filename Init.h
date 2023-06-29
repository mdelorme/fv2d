#pragma once

#include <fstream>

#include "SimInfo.h"
#include "BoundaryConditions.h"



namespace fv1d {

void init_sod_x(Array &Q) {
  std::cout << "Initializing SOD (x)" << std::endl;
  
  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      if (get_pos(i, j)[IX] <= 0.5) {
        Q[j][i][IR] = 1.0;
        Q[j][i][IP] = 1.0;
        Q[j][i][IU] = 0.0;
      }
      else {
        Q[j][i][IR] = 0.125;
        Q[j][i][IP] = 0.1;
        Q[j][i][IU] = 0.0;
      }
    }
  }
}

void init_sod_y(Array &Q) {
  std::cout << "Initializing SOD (y)" << std::endl;
  
  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      if (get_pos(i, j)[IY] <= 0.5) {
        Q[j][i][IR] = 1.0;
        Q[j][i][IP] = 1.0;
        Q[j][i][IU] = 0.0;
      }
      else {
        Q[j][i][IR] = 0.125;
        Q[j][i][IP] = 0.1;
        Q[j][i][IU] = 0.0;
      }
    }
  }
}

void init_blast(Array &Q) {
  std::cout << "Initializing blast" << std::endl;

  real_t xmid = 0.5 * (xmin+xmax);
  real_t ymid = 0.5 * (ymin+ymax);

  for (int j=jbeg; j < jend; ++j) {

    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);
      real_t x = pos[IX];
      real_t y = pos[IY];

      real_t xr = xmid - x;
      real_t yr = ymid - y;
      real_t r = sqrt(xr*xr+yr*yr);

      if (r < 0.2) {
        Q[j][i][IR] = 1.0;
        Q[j][i][IU] = 0.0;
        Q[j][i][IP] = 10.0;
      }
      else {
        Q[j][i][IR] = 1.2;
        Q[j][i][IU] = 0.0;
        Q[j][i][IP] = 0.1;
      }
    }
  }
}

void init_C91(Array &Q) {
  std::cout << "Initializing C91" << std::endl;

  srand(seed);

  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);
      real_t x = pos[IX];
      real_t y = pos[IY];

      real_t T = (1.0 + theta1*y);
      real_t rho = std::pow(T, m1);
      real_t prs = std::pow(T, m1+1.0);

      real_t pert = c91_pert * ((float)rand() / RAND_MAX * 2.0 - 1.0); 

      prs = prs * (1.0 + pert);

      Q[j][i][IR] = rho;
      Q[j][i][IU] = 0.0;
      Q[j][i][IV] = 0.0;
      Q[j][i][IP] = prs;
    }
  }
}

void init_diff(Array &Q) {
  std::cout << "Initializing Diffusion test" << std::endl;

  real_t xmid = 0.5 * (xmin + xmax);
  real_t ymid = 0.5 * (ymin + ymax);

  for (int j=jbeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);

      real_t x0 = (pos[IX]-xmid);
      real_t y0 = (pos[IY]-ymid);

      real_t r = sqrt(x0*x0+y0*y0);

      if (r < 0.2) 
        Q[j][i][IR] = 1.0;
      else
        Q[j][i][IR] = 0.1;

      Q[j][i][IP] = 1.0;
      Q[j][i][IU] = 1.0;
      Q[j][i][IV] = 0.0;
    }
  }
}

void init_rayleigh_taylor(Array &Q) {
 std::cout << "Initializing Rayleigh-Taylor" << std::endl;

  srand(seed);

  real_t ymid = 0.5*(ymin + ymax);

  for (int j=ibeg; j < jend; ++j) {
    for (int i=ibeg; i < iend; ++i) {
      Pos pos = get_pos(i, j);
      real_t x = pos[IX];
      real_t y = pos[IY];

      const real_t P0 = 2.5;

      if (y < ymid) {
        Q[j][i][IR] = 1.0;
        Q[j][i][IU] = 0.0;
        Q[j][i][IP] = P0 + 0.1 * g * y;
      }
      else {
        Q[j][i][IR] = 2.0;
        Q[j][i][IU] = 0.0;
        Q[j][i][IP] = P0 + 0.1 * g * y;
      }

      if (y > -1.0/3.0 && y < 1.0/3.0)
        Q[j][i][IV] = 0.01 * (1.0 + cos(4*M_PI*x)) * (1 + cos(3.0*M_PI*y))/4.0;
    }
  } 
}


void init(Array &Q) {
  if (problem == "sod_x")
    init_sod_x(Q);
  else if (problem == "sod_y")
    init_sod_y(Q);
  else if (problem == "blast")
    init_blast(Q);
  else if (problem == "C91")
    init_C91(Q);
  else if (problem == "rayleigh-taylor")
    init_rayleigh_taylor(Q);
  else if (problem == "diffusion")
    init_diff(Q);

  // One time init stuff
  init_boundaries();
}


}
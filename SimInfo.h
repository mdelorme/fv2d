#pragma once

#include <cmath>
#include <functional>
#include "INIReader.h"

namespace fv1d {

using real_t = double;
constexpr int Nfields = 4;
using Pos = std::array<real_t, 2>;
using State = std::array<real_t, Nfields>;
using Array = std::vector<std::vector<State>>;

enum IDir : uint8_t {
  IX = 0,
  IY = 1
};

enum IVar : uint8_t {
  IR = 0,
  IU = 1,
  IV = 2,
  IP = 3,
  IE = 3
};

enum RiemannSolver {
  HLL,
  HLLC
};

enum BoundaryType {
  BC_ABSORBING,
  BC_REFLECTING,
  BC_PERIODIC
};

enum ReconstructionType {
  PCM,
  PCM_WB,
  PLM
};

enum ThermalConductivityMode {
  TCM_CONSTANT,
  TCM_B02,
};

// Thermal conduction at boundary
enum BCTC_Mode {
  BCTC_NONE,              // Nothing special done
  BCTC_FIXED_TEMPERATURE, // Lock the temperature at the boundary
  BCTC_FIXED_GRADIENT     // Lock the gradient at the boundary
};

enum ViscosityMode {
  VSC_CONSTANT
};

// Run
real_t save_freq;
real_t tend;
std::string filename_out = "run.h5";
BoundaryType boundary_x = BC_REFLECTING;
BoundaryType boundary_y = BC_REFLECTING;
ReconstructionType reconstruction = PCM; 
RiemannSolver riemann_solver = HLL;
real_t CFL = 0.1;

// Mesh
int Nx;      // Number of domain cells
int Ny;      
int Ng;      // Number of ghosts
int Ntx;     // Total number of cells
int Nty;
int ibeg;    // First cell of the domain
int iend;    // First cell outside of the domain
int jbeg;
int jend;
real_t xmin; // Minimum boundary of the domain
real_t xmax; // Maximum boundary of the domain
real_t ymin;
real_t ymax;
real_t dx;   // Space step
real_t dy;

// Run and physics
real_t epsilon = 1.0e-6;
real_t gamma0 = 5.0/3.0;
bool gravity = false;
real_t g;
bool well_balanced_flux_at_y_bc = false;
bool well_balanced = false;
std::string problem;

// Thermal conduction
bool thermal_conductivity_active;
ThermalConductivityMode thermal_conductivity_mode;
real_t kappa;

BCTC_Mode bctc_ymin, bctc_ymax;
real_t bctc_ymin_value, bctc_ymax_value;

// Viscosity
bool viscosity_active;
ViscosityMode viscosity_mode;
real_t mu;

// Polytropes and such
real_t m1;
real_t theta1;
real_t m2;
real_t theta2;

// C91
real_t c91_pert;

// B02
real_t b02_ymid;
real_t b02_kappa1;
real_t b02_kappa2;
real_t b02_thickness;

// Misc 
int seed;
int log_frequency;

// Helper to get the position in the mesh
Pos get_pos(int i, int j) {
  return {xmin + (i-ibeg+0.5) * dx,
          ymin + (j-jbeg+0.5) * dy};
}


void read_inifile(std::string filename) {
  INIReader reader(filename);

  // Mesh
  Nx = reader.GetInteger("mesh", "Nx", 32);
  Ny = reader.GetInteger("mesh", "Ny", 32);
  Ng = reader.GetInteger("mesh", "Nghosts", 2);
  xmin = reader.GetFloat("mesh", "xmin", 0.0);
  xmax = reader.GetFloat("mesh", "xmax", 1.0);
  ymin = reader.GetFloat("mesh", "ymin", 0.0);
  ymax = reader.GetFloat("mesh", "ymax", 1.0);

  Ntx    = Nx + 2*Ng;
  Nty    = Ny + 2*Ng;
  ibeg = Ng;
  iend = Ng+Nx;
  jbeg = Ng;
  jend = Ng+Ny;

  dx = (xmax-xmin) / Nx;
  dy = (ymax-ymin) / Ny;

  // Run
  tend = reader.GetFloat("run", "tend", 1.0);
  save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  filename_out = reader.Get("run", "output_filename", "run.h5");

  std::string tmp;
  tmp = reader.Get("run", "boundaries_x", "reflecting");
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",         BC_REFLECTING},
    {"absorbing",          BC_ABSORBING},
    {"periodic",           BC_PERIODIC}
  };
  boundary_x = bc_map[tmp];
  tmp = reader.Get("run", "boundaries_y", "reflecting");
  boundary_y = bc_map[tmp];

  tmp = reader.Get("solvers", "reconstruction", "pcm");
  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",    PCM},
    {"pcm_wb", PCM_WB},
    {"plm",    PLM}
  };
  reconstruction = recons_map[tmp];

  std::cout << "Reconstruction is " << tmp << " (" << reconstruction << ")" << std::endl;

  tmp = reader.Get("solvers", "riemann_solver", "hllc");
  std::map<std::string, RiemannSolver> riemann_map{
    {"hll", HLL},
    {"hllc", HLLC}
  };
  riemann_solver = riemann_map[tmp];
  CFL = reader.GetFloat("solvers", "CFL", 0.8);

  // Physics
  epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
  gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
  gravity = reader.GetBoolean("physics", "gravity", false);
  g       = reader.GetFloat("physics", "g", 0.0);
  m1      = reader.GetFloat("polytrope", "m1", 1.0);
  theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
  m2      = reader.GetFloat("polytrope", "m2", 1.0);
  theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
  problem = reader.Get("physics", "problem", "blast");
  well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);

  // Thermal conductivity
  thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
  tmp = reader.Get("thermal_conduction", "conductivity_mode", "constant");
  std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
    {"constant", TCM_CONSTANT},
    {"B02",      TCM_B02}
  };
  thermal_conductivity_mode = thermal_conductivity_map[tmp];
  kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT}
  };
  tmp = reader.Get("thermal_conduction", "bc_xmin", "none");
  bctc_ymin = bctc_map[tmp];
  tmp = reader.Get("thermal_conduction", "bc_xmax", "none");
  bctc_ymax = bctc_map[tmp];
  bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_xmin_value", 1.0);
  bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_xmax_value", 1.0);

  // Viscosity
  viscosity_active = reader.GetBoolean("viscosity", "active", false);
  tmp = reader.Get("viscosity", "viscosity_mode", "constant");
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  viscosity_mode = viscosity_map[tmp];
  mu = reader.GetFloat("viscosity", "mu", 0.0);

  // C91
  c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

  // Misc
  seed = reader.GetInteger("misc", "seed", 12345);
  log_frequency = reader.GetInteger("misc", "log_frequency", 10);
} 

void allocate_array(Array &arr) {
  arr.resize(Nty);
  for (auto &a : arr)
    a.resize(Ntx);
}

void copy_array(Array &dst, Array &src) {
  for (int j=0; j < Nty; ++j)
    for (int i=0; i < Ntx; ++i)
      dst[j][i] = src[j][i];
}

// Function to apply for the boundaries
std::function<State(Array&, int, int, int, int, real_t, IDir)> bc_function_x;
std::function<State(Array&, int, int, int, int, real_t, IDir)> bc_function_y;
}

// All states operations
#include "States.h"

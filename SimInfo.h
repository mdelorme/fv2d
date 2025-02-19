#pragma once

#include <cmath>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>

namespace fv2d {

using real_t = double;

#ifdef MHD
constexpr int Nfields = 8;
#else
constexpr int Nfields = 4;
#endif

using Pos   = Kokkos::Array<real_t, 2>;
using State = Kokkos::Array<real_t, Nfields>;
using Array = Kokkos::View<real_t***>;
using ParallelRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

struct RestartInfo {
  real_t time;
  int iteration;
};

enum IDir : uint8_t {
  IX = 0,
  IY = 1
};

#ifdef MHD
enum IVar : uint8_t {
  IR = 0,
  IU = 1,
  IV = 2,
  IW = 3,
  IP = 4,
  IE = 4,
  IBX = 5,
  IBY = 6,
  IBZ = 7
};
#else
enum IVar : uint8_t {
  IR = 0,
  IU = 1,
  IV = 2,
  IP = 3,
  IE = 3
};
#endif

enum RiemannSolver {
  HLL,
  HLLC,
  HLLD
};

enum BoundaryType {
  BC_ABSORBING,
  BC_REFLECTING,
  BC_PERIODIC
};

enum TimeStepping {
  TS_EULER,
  TS_RK2
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
struct Params {
  real_t save_freq;
  real_t tend;
  std::string filename_out = "run";
  std::string restart_file = "";
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  TimeStepping time_stepping = TS_EULER;
  real_t CFL = 0.1;

  bool multiple_outputs = false;

  // Parallel stuff
  ParallelRange range_tot;
  ParallelRange range_dom;
  ParallelRange range_xbound;
  ParallelRange range_ybound;
  ParallelRange range_slopes;

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

  // H84
  real_t h84_pert;

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
};

// Helper to get the position in the mesh
KOKKOS_INLINE_FUNCTION
Pos getPos(const Params& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

Params readInifile(std::string filename) {
  INIReader reader(filename);

  Params res;

  // Mesh
  res.Nx = reader.GetInteger("mesh", "Nx", 32);
  res.Ny = reader.GetInteger("mesh", "Ny", 32);
  res.Ng = reader.GetInteger("mesh", "Nghosts", 2);
  res.xmin = reader.GetFloat("mesh", "xmin", 0.0);
  res.xmax = reader.GetFloat("mesh", "xmax", 1.0);
  res.ymin = reader.GetFloat("mesh", "ymin", 0.0);
  res.ymax = reader.GetFloat("mesh", "ymax", 1.0);

  res.Ntx  = res.Nx + 2*res.Ng;
  res.Nty  = res.Ny + 2*res.Ng;
  res.ibeg = res.Ng;
  res.iend = res.Ng+res.Nx;
  res.jbeg = res.Ng;
  res.jend = res.Ng+res.Ny;

  res.dx = (res.xmax-res.xmin) / res.Nx;
  res.dy = (res.ymax-res.ymin) / res.Ny;

  // Run
  res.tend = reader.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = reader.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = reader.Get("run", "restart_file", "");
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
    
  res.save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = reader.Get("run", "output_filename", "run");

  std::string tmp;
  tmp = reader.Get("run", "boundaries_x", "reflecting");
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",         BC_REFLECTING},
    {"absorbing",          BC_ABSORBING},
    {"periodic",           BC_PERIODIC}
  };
  res.boundary_x = bc_map[tmp];
  tmp = reader.Get("run", "boundaries_y", "reflecting");
  res.boundary_y = bc_map[tmp];

  tmp = reader.Get("solvers", "reconstruction", "pcm");
  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",    PCM},
    {"pcm_wb", PCM_WB},
    {"plm",    PLM}
  };
  res.reconstruction = recons_map[tmp];

  tmp = reader.Get("solvers", "riemann_solver", "hllc");
  std::map<std::string, RiemannSolver> riemann_map{
    {"hll", HLL},
    {"hllc", HLLC}
  };
  res.riemann_solver = riemann_map[tmp];

  tmp = reader.Get("solvers", "time_stepping", "euler");
  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = ts_map[tmp];

  res.CFL = reader.GetFloat("solvers", "CFL", 0.8);

  // Physics
  res.epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
  res.gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
  res.gravity = reader.GetBoolean("physics", "gravity", false);
  res.g       = reader.GetFloat("physics", "g", 0.0);
  res.m1      = reader.GetFloat("polytrope", "m1", 1.0);
  res.theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
  res.m2      = reader.GetFloat("polytrope", "m2", 1.0);
  res.theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
  res.problem = reader.Get("physics", "problem", "blast");
  res.well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);

  // Thermal conductivity
  res.thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
  tmp = reader.Get("thermal_conduction", "conductivity_mode", "constant");
  std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
    {"constant", TCM_CONSTANT},
    {"B02",      TCM_B02}
  };
  res.thermal_conductivity_mode = thermal_conductivity_map[tmp];
  res.kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT}
  };
  tmp = reader.Get("thermal_conduction", "bc_xmin", "none");
  res.bctc_ymin = bctc_map[tmp];
  tmp = reader.Get("thermal_conduction", "bc_xmax", "none");
  res.bctc_ymax = bctc_map[tmp];
  res.bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_xmin_value", 1.0);
  res.bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_xmax_value", 1.0);

  // Viscosity
  res.viscosity_active = reader.GetBoolean("viscosity", "active", false);
  tmp = reader.Get("viscosity", "viscosity_mode", "constant");
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  res.viscosity_mode = viscosity_map[tmp];
  res.mu = reader.GetFloat("viscosity", "mu", 0.0);

  // H84
  res.h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

  // C91
  res.c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

  // Misc
  res.seed = reader.GetInteger("misc", "seed", 12345);
  res.log_frequency = reader.GetInteger("misc", "log_frequency", 10);


  // Parallel ranges
  res.range_tot = ParallelRange({0, 0}, {res.Ntx, res.Nty});
  res.range_dom = ParallelRange({res.ibeg, res.jbeg}, {res.iend, res.jend});
  res.range_xbound = ParallelRange({0, res.jbeg}, {res.Ng, res.jend});
  res.range_ybound = ParallelRange({0, 0}, {res.Ntx, res.Ng});
  res.range_slopes = ParallelRange({res.ibeg-1, res.jbeg-1}, {res.iend+1, res.jend+1});

  return res;
} 
}

// All states operations
#include "States.h"

namespace fv2d {
void consToPrim(Array U, Array Q, const Params &params) {
  Kokkos::parallel_for( "Conservative to Primitive", 
                        params.range_tot,
                        KOKKOS_LAMBDA(const int i, const int j) {
                          State Uloc = getStateFromArray(U, i, j);
                          State Qloc = consToPrim(Uloc, params);
                          setStateInArray(Q, i, j, Qloc);
                        });
}

void primToCons(Array &Q, Array &U, const Params &params) {
  Kokkos::parallel_for( "Primitive to Conservative", 
                        params.range_tot,
                        KOKKOS_LAMBDA(const int i, const int j) {
                          State Qloc = getStateFromArray(Q, i, j);
                          State Uloc = primToCons(Qloc, params);
                          setStateInArray(U, i, j, Uloc);
                        });
}

}

#pragma once

#include <cmath>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>

namespace fv2d {

using real_t = double;
constexpr int Nfields = 4;
using Pos   = Kokkos::Array<real_t, 2>;
using State = Kokkos::Array<real_t, Nfields>;
using Array = Kokkos::View<real_t***>;
using ParallelRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

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
  BC_PERIODIC,
  BC_TRILAYER_DAMPING
};

enum TimeStepping {
  TS_EULER,
  TS_RK2
};

enum ReconstructionType {
  PCM,
  PCM_WB,
  PCM_WB2,
  PLM
};

enum ThermalConductivityMode {
  TCM_CONSTANT,
  TCM_B02,
  TCM_C2020_TRI,
  TCM_ISO3
};

enum HeatingMode {
  HM_C2020,
  HM_C2020_TRI,
  HM_COOLING_ISO,
};

// Thermal conduction at boundary
enum BCTC_Mode {
  BCTC_NONE,              // Nothing special done
  BCTC_FIXED_TEMPERATURE, // Lock the temperature at the boundary
  BCTC_FIXED_GRADIENT,    // Lock the gradient at the boundary
  BCTC_NO_FLUX,           // Thermal Flux is set to 0 at the boundary
  BCTC_NO_CONDUCTION,     // Top and bottom flux of cell are matched
};

enum ViscosityMode {
  VSC_CONSTANT
};

// Run
struct Params {
  real_t save_freq;
  real_t tend;
  std::string filename_out = "run.h5";
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  TimeStepping time_stepping = TS_EULER;
  real_t CFL = 0.1;

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

  // Heating
  bool heating_active;
  HeatingMode heating_mode;
  bool log_total_heating;

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

  // Currie 2020
  real_t c20_H;
  real_t c20_heating_fac;

  // Tri-Layer
  real_t tri_y1, tri_y2;
  real_t tri_pert;
  real_t tri_k1, tri_k2;
  real_t T0, rho0;

  // Isothermal triple layer
  real_t iso3_dy0, iso3_dy1, iso3_dy2;
  real_t iso3_theta1, iso3_theta2;
  real_t iso3_m1, iso3_m2;
  real_t iso3_pert;
  real_t iso3_k1, iso3_k2;
  real_t iso3_T0, iso3_rho0;

  // Misc 
  int seed;
  int log_frequency;
  bool log_energy_contributions;
  int log_energy_frequency;
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
  res.save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = reader.Get("run", "output_filename", "run.h5");

  std::string tmp;
  tmp = reader.Get("run", "boundaries_x", "reflecting");
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",         BC_REFLECTING},
    {"absorbing",          BC_ABSORBING},
    {"periodic",           BC_PERIODIC},
    {"tri_layer_damping",  BC_TRILAYER_DAMPING}
  };
  res.boundary_x = bc_map[tmp];
  tmp = reader.Get("run", "boundaries_y", "reflecting");
  res.boundary_y = bc_map[tmp];

  tmp = reader.Get("solvers", "reconstruction", "pcm");
  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",     PCM},
    {"pcm_wb",  PCM_WB},
    {"pcm_wb2", PCM_WB2},
    {"plm",     PLM}
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
    {"constant",  TCM_CONSTANT},
    {"B02",       TCM_B02},
    {"tri_layer", TCM_C2020_TRI},
    {"iso-three", TCM_ISO3}
  };
  res.thermal_conductivity_mode = thermal_conductivity_map[tmp];
  res.kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT},
    {"no_flux",           BCTC_NO_FLUX},
    {"no_conduction",     BCTC_NO_CONDUCTION},
  };
  tmp = reader.Get("thermal_conduction", "bc_ymin", "none");
  res.bctc_ymin = bctc_map[tmp];
  tmp = reader.Get("thermal_conduction", "bc_ymax", "none");
  res.bctc_ymax = bctc_map[tmp];
  res.bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_ymin_value", 1.0);
  res.bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_ymax_value", 1.0);  

  // Viscosity
  res.viscosity_active = reader.GetBoolean("viscosity", "active", false);
  tmp = reader.Get("viscosity", "viscosity_mode", "constant");
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  res.viscosity_mode = viscosity_map[tmp];
  res.mu = reader.GetFloat("viscosity", "mu", 0.0);

  // Heating function 
  res.heating_active = reader.GetBoolean("heating", "active", false);
  tmp = reader.Get("heating", "mode", "C2020");
  std::map<std::string, HeatingMode> heating_map{
    {"C2020", HM_C2020},
    {"tri_layer", HM_C2020_TRI},
    {"isothermal_cooling", HM_COOLING_ISO}
  };
  res.heating_mode = heating_map[tmp];
  res.log_total_heating = reader.GetBoolean("misc", "log_total_heating", false);

  // H84
  res.h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

  // C91
  res.c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

  // C20
  res.c20_H = reader.GetFloat("C20", "H", 0.2);
  res.c20_heating_fac = reader.GetFloat("C20", "heating_fac", 2.0);

  // Tri-layer
  res.tri_y1   = reader.GetFloat("tri_layer", "y1", 1.0);
  res.tri_y2   = reader.GetFloat("tri_layer", "y2", 2.0);
  res.tri_pert = reader.GetFloat("tri_layer", "perturbation", 1.0e-3);
  res.tri_k1   = reader.GetFloat("tri_layer", "kappa1", 0.07);
  res.tri_k2   = reader.GetFloat("tri_layer", "kappa2", 1.5);
  res.T0       = reader.GetFloat("tri_layer", "T0", 1.0);
  res.rho0     = reader.GetFloat("tri_layer", "rho0", 1.0);

  // Isothermal triple layer
  res.iso3_dy0    = reader.GetFloat("iso_three_layer", "dy0", 1.0);
  res.iso3_dy1    = reader.GetFloat("iso_three_layer", "dy1", 2.0);
  res.iso3_dy2    = reader.GetFloat("iso_three_layer", "dy2", 2.0);
  res.iso3_theta1 = reader.GetFloat("iso_three_layer", "theta1", 2.0);
  res.iso3_theta2 = reader.GetFloat("iso_three_layer", "theta2", 2.0);
  res.iso3_pert   = reader.GetFloat("iso_three_layer", "perturbation", 1.0e-3);
  res.iso3_k1     = reader.GetFloat("iso_three_layer", "k1", 0.07);
  res.iso3_k2     = reader.GetFloat("iso_three_layer", "k2", 1.5);
  res.iso3_m1     = reader.GetFloat("iso_three_layer", "m1", 1.0);
  res.iso3_m2     = reader.GetFloat("iso_three_layer", "m2", 1.0);
  res.iso3_T0     = reader.GetFloat("iso_three_layer", "T0", 1.0);
  res.iso3_rho0   = reader.GetFloat("iso_three_layer", "rho0", 1.0);


  // Misc
  res.seed = reader.GetInteger("misc", "seed", 12345);
  res.log_frequency = reader.GetInteger("misc", "log_frequency", 10);
  res.log_energy_contributions = reader.GetBoolean("misc", "log_energy_contributions", false);
  res.log_energy_frequency = reader.GetFloat("misc", "log_energy_frequency", 10);


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

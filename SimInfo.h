#pragma once

namespace fv2d { using real_t = double; }

#include <cassert>
#include <cmath>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>
#include "Spline.h"

namespace fv2d {
constexpr int Nfields = 4;
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

enum ISide : uint8_t {
  ILEFT  = 0,
  IRIGHT = 1
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

enum FluxSolver {
  FLUX_CTU,
  FLUX_GODOUNOV
};

enum BoundaryType {
  BC_ABSORBING,
  BC_REFLECTING,
  BC_PERIODIC,
  BC_RADIAL_REFLECTING,
  BC_FIXED_READFILE,
};

enum TimeStepping {
  TS_EULER,
  TS_RK2
};

enum ReconstructionType {
  // 1d reconstruction
  PCM,
  PCM_WB,
  PLM,

  // 2d reconstruction (gradient)
  RECONS_NAIVE,
  RECONS_BRUNER,
  RECONS_BJ,
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

// Geometry
enum GeometryType { 
  GEO_CARTESIAN,
  GEO_RADIAL,    // Geometry is radial (cf: Calhoun 2008)
  GEO_RADIAL_CONVEX,
  GEO_COLELLA,
  GEO_RING,
  GEO_TEST,
};

enum Mapc2pType { 
  MAP_CENTROID,
  MAP_CENTER,
  MAP_MAPPED,
};

enum GravityType { 
  GRAV_NONE,
  GRAV_CONST,
  GRAV_READFILE,
};

enum GradientType { 
  LEAST_SQUARE,
  LEAST_SQUARE_WIDE,
  LEAST_SQUARE_NODE,
  GREEN_GAUSS,
  GREEN_GAUSS_WIDE,
};

// Pos arithmetic

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
KOKKOS_INLINE_FUNCTION
const Pos operator/(const Pos &p, real_t f)
{
  return {p[IX] / f,
          p[IY] / f};
}

// Run
struct Params {
  real_t save_freq;
  real_t tend;
  std::string path_out = ".";
  std::string filename_out = "run";
  std::string restart_file = "";
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  bool reflective_flux=false;
  bool reflective_flux_wb=false;
  bool fixed_spline_boundaries=false;
  ReconstructionType reconstruction = PCM; 
  bool hancock_ts = false; 
  RiemannSolver riemann_solver = HLL;
  TimeStepping time_stepping = TS_EULER;
  FluxSolver flux_solver = FLUX_GODOUNOV;
  real_t CFL = 0.1;

  bool multiple_outputs = false;

  // Parallel stuff
  ParallelRange range_tot;
  ParallelRange range_dom;
  ParallelRange range_xbound;
  ParallelRange range_ybound;
  ParallelRange range_slopes;
  ParallelRange range_fluxes;

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

  // Geometry
  GeometryType geometry_type = GEO_CARTESIAN;
  real_t geometry_colella_param = 0.6 / (2.0 * M_PI);
  real_t radial_radius = 1.0;
  Mapc2pType mapc2p_type = MAP_CENTROID;

  // Run and physics
  real_t epsilon = 1.0e-6;
  real_t gamma0 = 5.0/3.0;
  GravityType gravity = GRAV_NONE;
  real_t g;
  bool well_balanced_flux_at_y_bc = false;
  bool well_balanced = false;
  std::string problem;

  // Gradient
  GradientType gradient_type;
  bool use_pressure_gradient; // resolve pressure gradient ouside of the riemann (does not work well)

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

  // Splines
  Spline spl_rho, spl_prs, spl_grav;
  real_t pert;

  /////////////////////////////////////////////////////////
  // ring test init
  real_t ring_velocity;
  real_t ring_p_in;
  real_t ring_rho_in;
  real_t ring_p_out;
  real_t ring_rho_out;
  bool ring_scale_vel_r;
  int ring_init_type;
  real_t init_type2_radius;
};

// Helper to get the position in the mesh
KOKKOS_INLINE_FUNCTION
Pos getPos(const Params& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

Params readInifile(std::string filename) {
  INIReader reader(filename);

  auto read_map = [&reader](const auto& map, const std::string& section, const std::string& name, const std::string& default_value){
    std::string tmp;
    tmp = reader.Get(section, name, default_value);

    if (map.count(tmp) == 0) {
      tmp = "\nautorized values: ";
      for (auto elem : map) tmp += elem.first + ", ";
      throw std::runtime_error(std::string("bad parameter for ") + name + ": " + tmp);
    }
    return map.at(tmp);
  };

  Params res;
  std::map<std::string, Mapc2pType> mapc2p_map2{
    {"centroid", MAP_CENTROID},
    {"center",   MAP_CENTER},
    {"mapped",   MAP_MAPPED},
  };
  res.mapc2p_type = read_map(mapc2p_map2, "mesh", "mapc2p_type", "centroid");

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
  res.path_out = reader.Get("run", "output_path", ".");
  
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",         BC_REFLECTING},
    {"absorbing",          BC_ABSORBING},
    {"periodic",           BC_PERIODIC},
    {"radial_reflecting",  BC_RADIAL_REFLECTING},
    {"fixed_readfile",     BC_FIXED_READFILE},
  };
  res.boundary_x = read_map(bc_map, "run", "boundaries_x", "reflecting");
  res.boundary_y = read_map(bc_map, "run", "boundaries_y", "reflecting");

  res.reflective_flux = reader.GetBoolean("run", "reflective_flux", false); 
  res.reflective_flux_wb = reader.GetBoolean("run", "reflective_flux_wb", false); 
  res.fixed_spline_boundaries = reader.GetBoolean("run", "fixed_spline_boundaries", false); 

  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",          PCM},
    {"pcm_wb",       PCM_WB},
    {"plm",          PLM},

    {"grad-naive",   RECONS_NAIVE},
    {"grad-bruner",  RECONS_BRUNER},
    {"grad-bj",      RECONS_BJ},
  };
  res.reconstruction = read_map(recons_map, "solvers", "reconstruction", "pcm");

  res.hancock_ts = reader.GetBoolean("solvers", "hancock", false);

  std::map<std::string, RiemannSolver> riemann_map{
    {"hll", HLL},
    {"hllc", HLLC}
  };
  res.riemann_solver = read_map(riemann_map, "solvers", "riemann_solver", "hllc");

  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = read_map(ts_map, "solvers", "time_stepping", "euler");

  std::map<std::string, FluxSolver> flux_solver_map{
    {"ctu",      FLUX_CTU},
    {"godounov", FLUX_GODOUNOV}
  };
  res.flux_solver = read_map(flux_solver_map, "solvers", "flux_solver", "godounov");

  res.CFL = reader.GetFloat("solvers", "CFL", 0.8);

  // Geometry
  std::map<std::string, GeometryType> geo_map{
    {"cartesian",      GEO_CARTESIAN},
    {"radial",         GEO_RADIAL},
    {"radial_convex",  GEO_RADIAL_CONVEX},
    {"colella",        GEO_COLELLA},
    {"ring",           GEO_RING},
    {"maptest",        GEO_TEST},
  };
  res.geometry_type = read_map(geo_map, "mesh", "geometry", "cartesian");
  res.geometry_colella_param = reader.GetFloat("mesh", "geometry_colella_param", 0.6 / (2.0 * M_PI));
  res.radial_radius = reader.GetFloat("mesh", "radial_radius", 1.0);

  std::map<std::string, Mapc2pType> mapc2p_map{
    {"centroid", MAP_CENTROID},
    {"center",   MAP_CENTER},
    {"mapped",   MAP_MAPPED},
  };
  res.mapc2p_type = read_map(mapc2p_map, "mesh", "mapc2p_type", "centroid");
  
  // Physics
  res.epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
  res.gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
  res.g       = reader.GetFloat("physics", "g", 0.0);
  res.m1      = reader.GetFloat("polytrope", "m1", 1.0);
  res.theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
  res.m2      = reader.GetFloat("polytrope", "m2", 1.0);
  res.theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
  res.problem = reader.Get("physics", "problem", "blast");
  res.well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);

  std::map<std::string, GravityType> gravity_map{
    {"none",     GRAV_NONE},
    {"false",    GRAV_NONE},
    {"const",    GRAV_CONST},
    {"true",     GRAV_CONST},
    {"readfile", GRAV_READFILE},
  };
  res.gravity = read_map(gravity_map, "physics", "gravity", "false");

  std::map<std::string, GradientType> gradient_map{
    {"least-square",       LEAST_SQUARE},
    {"least-square-wide",  LEAST_SQUARE_WIDE},
    {"least-square-node",  LEAST_SQUARE_NODE},
    {"green-gauss",        GREEN_GAUSS},
    {"green-gauss-wide",   GREEN_GAUSS_WIDE},
  };
  res.gradient_type = read_map(gradient_map, "solvers", "gradient", "least-square");

  res.use_pressure_gradient = reader.GetBoolean("solvers", "use_pressure_gradient", false);

  // Thermal conductivity
  res.thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
  std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
    {"constant", TCM_CONSTANT},
    {"B02",      TCM_B02}
  };
  res.thermal_conductivity_mode = read_map(thermal_conductivity_map, "thermal_conduction", "conductivity_mode", "constant");
  res.kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT}
  };
  res.bctc_ymin = read_map(bctc_map, "thermal_conduction", "bc_xmin", "none");
  res.bctc_ymax = read_map(bctc_map, "thermal_conduction", "bc_xmax", "none");
  res.bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_xmin_value", 1.0);
  res.bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_xmax_value", 1.0);

  // Viscosity
  res.viscosity_active = reader.GetBoolean("viscosity", "active", false);
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  res.viscosity_mode = read_map(viscosity_map, "viscosity", "viscosity_mode", "constant");
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
  res.range_fluxes = ParallelRange({res.ibeg-1, res.jbeg-1}, {res.iend+1, res.jend+1});

  // Spline
  std::string spline_data_path = reader.Get("physics", "spline_data", "spline.data");
  if (res.problem == "readfile") {
    res.spl_rho  = Spline(spline_data_path, Spline::OF_RHO);
    res.spl_prs  = Spline(spline_data_path, Spline::OF_PRS);
  }
  if (res.gravity == GRAV_READFILE)
    res.spl_grav = Spline(spline_data_path, Spline::OF_GRAVITY);

  res.pert = reader.GetFloat("physics", "perturbation", 1.0e-3);


  /////////////////////////////////////////////////////////////////
  
  // ring test init
  res.ring_velocity     = reader.GetFloat("ring", "velocity", 0.1);
  res.ring_p_in         = reader.GetFloat("ring", "prs_in", 1.0);
  res.ring_rho_in       = reader.GetFloat("ring", "rho_in", 1.0);
  res.ring_p_out        = reader.GetFloat("ring", "prs_out", 1.0);
  res.ring_rho_out      = reader.GetFloat("ring", "rho_out", 1.0);
  res.ring_scale_vel_r  = reader.GetBoolean("ring", "scale_vel_radius", true);
  res.ring_init_type    = reader.GetInteger("ring", "init_type", 1);
  res.init_type2_radius = reader.GetFloat("ring", "init_type2_radius", 0.125);
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

void checkNegatives(Array &Q, const Params &params) {
  uint64_t negative_density  = 0;
  uint64_t negative_pressure = 0;
  uint64_t nan_count = 0;

  Kokkos::parallel_reduce(
    "Check negative density/pressure", 
    params.range_dom,
    KOKKOS_LAMBDA(const int i, const int j, uint64_t& lnegative_density, uint64_t& lnegative_pressure, uint64_t& lnan_count) {
      constexpr real_t eps = 1.0e-6;
      if (Q(j, i, IR) < eps) {
        Q(j, i, IR) = eps;
        lnegative_density++;
      }
      if (Q(j, i, IP) < eps) {
        Q(j, i, IP) = eps;
        lnegative_pressure++;
      }

      for (int ivar=0; ivar < Nfields; ++ivar)
        if (std::isnan(Q(j, i, ivar)))
          lnan_count++;

    }, negative_density, negative_pressure, nan_count);

    if (negative_density) 
      std::cout << "--> negative density: " << negative_density << std::endl;
    if (negative_pressure)
      std::cout << "--> negative pressure: " << negative_pressure << std::endl;
    if (nan_count)
      std::cout << "--> NaN detected." << std::endl;
}


}

#include "Gradient.h"


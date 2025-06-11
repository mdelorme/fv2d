#pragma once

namespace fv2d { using real_t = double; }

#include <cassert>
#include <cmath>
#include <string>
#include <iomanip>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>
#include "Spline.h"

namespace fv2d {

namespace {
  auto read_map(INIReader &reader, const auto& map, const std::string& section, const std::string& name, const std::string& default_value){
    std::string tmp;
    tmp = reader.Get(section, name, default_value);

    if (map.count(tmp) == 0) {
      tmp = "\nallowed values: ";
      for (auto elem : map) tmp += elem.first + ", ";
      throw std::runtime_error(std::string("bad parameter for ") + name + ": " + tmp);
    }
    return map.at(tmp);
  };
}

using real_t = double;
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
  BC_ISOTHERMAL_DIRICHLET,
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
  GRAV_RADIAL_LIN,
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

// All parameters that should be copied on the device
struct DeviceParams { 
  // Thermodynamics
  real_t gamma0 = 5.0/3.0;
  
  // Gravity
  GravityType gravity = GRAV_NONE;
  real_t g;
  bool well_balanced_flux_at_y_bc = false;
  bool well_balanced = false;
  
  // Thermal conductivity
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
  
  // Boundaries
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;

  // impose flux at boundaries
  bool reflective_flux=false;
  bool reflective_flux_wb=false;
  bool fixed_spline_boundaries=false;

  // Godunov
  bool hancock_ts = false; 
  
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  TimeStepping time_stepping = TS_EULER;
  FluxSolver flux_solver = FLUX_GODOUNOV;
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

  // Geometry
  GeometryType geometry_type = GEO_CARTESIAN;
  real_t geometry_colella_param = 0.6 / (2.0 * M_PI);
  real_t radial_radius = 1.0;
  Mapc2pType mapc2p_type = MAP_CENTROID;

  // Coriolis terms
  bool coriolis_active = false;
  real_t coriolis_omega;

  // Gradient
  GradientType gradient_type;
  bool use_pressure_gradient; // resolve pressure gradient ouside of the riemann (does not work well)


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

  // Misc stuff
  real_t epsilon = 1.0e-6;
  
  void init_from_inifile(INIReader &reader) {

    std::map<std::string, FluxSolver> flux_solver_map{
      {"ctu",      FLUX_CTU},
      {"godounov", FLUX_GODOUNOV}
    };
    flux_solver = read_map(reader, flux_solver_map, "solvers", "flux_solver", "godounov");

    // Geometry
    std::map<std::string, GeometryType> geo_map{
      {"cartesian",      GEO_CARTESIAN},
      {"radial",         GEO_RADIAL},
      {"radial_convex",  GEO_RADIAL_CONVEX},
      {"colella",        GEO_COLELLA},
      {"ring",           GEO_RING},
      {"maptest",        GEO_TEST},
    };
    geometry_type = read_map(reader, geo_map, "mesh", "geometry", "cartesian");
    geometry_colella_param = reader.GetFloat("mesh", "geometry_colella_param", 0.6 / (2.0 * M_PI));
    radial_radius = reader.GetFloat("mesh", "radial_radius", 1.0);

    std::map<std::string, Mapc2pType> mapc2p_center_type{
      {"centroid", MAP_CENTROID},
      {"center",   MAP_CENTER},
      {"mapped",   MAP_MAPPED},
    };
    mapc2p_type = read_map(reader, mapc2p_center_type, "mesh", "mapc2p_type", "centroid");

    std::map<std::string, GravityType> gravity_map{
      {"none",          GRAV_NONE},
      {"false",         GRAV_NONE},
      {"const",         GRAV_CONST},
      {"true",          GRAV_CONST},
      {"radial_linear", GRAV_RADIAL_LIN},
      {"readfile",      GRAV_READFILE},
    };
    gravity = read_map(reader, gravity_map, "physics", "gravity", "false");

    coriolis_active = reader.GetBoolean("coriolis", "active", false);
    coriolis_omega = reader.GetFloat("coriolis", "omega", 2.8329587910568913e-06);

    std::map<std::string, GradientType> gradient_map{
      {"least-square",       LEAST_SQUARE},
      {"least-square-wide",  LEAST_SQUARE_WIDE},
      {"least-square-node",  LEAST_SQUARE_NODE},
      {"green-gauss",        GREEN_GAUSS},
      {"green-gauss-wide",   GREEN_GAUSS_WIDE},
    };
    gradient_type = read_map(reader, gradient_map, "solvers", "gradient", "least-square");
    use_pressure_gradient = reader.GetBoolean("solvers", "use_pressure_gradient", false);

    // Mesh
    Nx = reader.GetInteger("mesh", "Nx", 32);
    Ny = reader.GetInteger("mesh", "Ny", 32);
    Ng = reader.GetInteger("mesh", "Nghosts", 2);
    xmin = reader.GetFloat("mesh", "xmin", 0.0);
    xmax = reader.GetFloat("mesh", "xmax", 1.0);
    ymin = reader.GetFloat("mesh", "ymin", 0.0);
    ymax = reader.GetFloat("mesh", "ymax", 1.0);
    
    Ntx  = Nx + 2*Ng;
    Nty  = Ny + 2*Ng;
    ibeg = Ng;
    iend = Ng+Nx;
    jbeg = Ng;
    jend = Ng+Ny;
    
    dx = (xmax-xmin) / Nx;
    dy = (ymax-ymin) / Ny;
    
    CFL = reader.GetFloat("solvers", "CFL", 0.8);
    std::map<std::string, BoundaryType> bc_map{
      {"reflecting",            BC_REFLECTING},
      {"absorbing",             BC_ABSORBING},
      {"periodic",              BC_PERIODIC},
      {"radial_reflecting",     BC_RADIAL_REFLECTING},
      {"fixed_readfile",        BC_FIXED_READFILE},
      {"isothermal_dirichlet",  BC_ISOTHERMAL_DIRICHLET},
    };
    boundary_x = read_map(reader, bc_map, "run", "boundaries_x", "reflecting");
    boundary_y = read_map(reader, bc_map, "run", "boundaries_y", "reflecting");

    std::map<std::string, ReconstructionType> recons_map{
      {"pcm",          PCM},
      {"pcm_wb",       PCM_WB},
      {"plm",          PLM},

      {"grad-naive",   RECONS_NAIVE},
      {"grad-bruner",  RECONS_BRUNER},
      {"grad-bj",      RECONS_BJ},
    };
    reconstruction = read_map(reader, recons_map, "solvers", "reconstruction", "pcm");

    std::map<std::string, RiemannSolver> riemann_map{
      {"hll", HLL},
      {"hllc", HLLC}
    };
    riemann_solver = read_map(reader, riemann_map, "solvers", "riemann_solver", "hllc");

    // Physics
    epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
    gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
    g       = reader.GetFloat("physics", "g", 0.0);
    m1      = reader.GetFloat("polytrope", "m1", 1.0);
    theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
    m2      = reader.GetFloat("polytrope", "m2", 1.0);
    theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
    well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);

    // Thermal conductivity
    thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
    std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
      {"constant", TCM_CONSTANT},
      {"B02",      TCM_B02}
    };
    thermal_conductivity_mode = read_map(reader, thermal_conductivity_map, "thermal_conduction", "conductivity_mode", "constant");
    kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

    std::map<std::string, BCTC_Mode> bctc_map{
      {"none",              BCTC_NONE},
      {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
      {"fixed_gradient",    BCTC_FIXED_GRADIENT}
    };
    bctc_ymin = read_map(reader, bctc_map, "thermal_conduction", "bc_xmin", "none");
    bctc_ymax = read_map(reader, bctc_map, "thermal_conduction", "bc_xmax", "none");
    bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_xmin_value", 1.0);
    bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_xmax_value", 1.0);

    // Viscosity
    viscosity_active = reader.GetBoolean("viscosity", "active", false);
    std::map<std::string, ViscosityMode> viscosity_map{
      {"constant", VSC_CONSTANT},
    };
    viscosity_mode = read_map(reader, viscosity_map, "viscosity", "viscosity_mode", "constant");
    mu = reader.GetFloat("viscosity", "mu", 0.0);

    // H84
    h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

    // C91
    c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

    reflective_flux = reader.GetBoolean("run", "reflective_flux", false); 
    reflective_flux_wb = reader.GetBoolean("run", "reflective_flux_wb", false); 
    fixed_spline_boundaries = reader.GetBoolean("run", "fixed_spline_boundaries", false); 
    hancock_ts = reader.GetBoolean("solvers", "hancock", false);


    // Spline
    std::string spline_data_path = reader.Get("physics", "spline_data", "spline.data");
    if (reader.Get("physics", "problem", "unknown") == "readfile") {
      spl_rho  = Spline(spline_data_path, Spline::OF_RHO);
      spl_prs  = Spline(spline_data_path, Spline::OF_PRS);
    }
    if (gravity == GRAV_READFILE)
      spl_grav = Spline(spline_data_path, Spline::OF_GRAVITY);
    
    pert = reader.GetFloat("physics", "perturbation", 1.0e-3);
    
    // ring test init
    ring_velocity     = reader.GetFloat("ring", "velocity", 0.1);
    ring_p_in         = reader.GetFloat("ring", "prs_in", 1.0);
    ring_rho_in       = reader.GetFloat("ring", "rho_in", 1.0);
    ring_p_out        = reader.GetFloat("ring", "prs_out", 1.0);
    ring_rho_out      = reader.GetFloat("ring", "rho_out", 1.0);
    ring_scale_vel_r  = reader.GetBoolean("ring", "scale_vel_radius", true);
    ring_init_type    = reader.GetInteger("ring", "init_type", 1);
    init_type2_radius = reader.GetFloat("ring", "init_type2_radius", 0.125);
  }
};

// All the parameters
struct Params {
  real_t save_freq;
  real_t tend;
  INIReader reader;
  std::string path_out = ".";
  std::string filename_out = "run";
  std::string restart_file = "";
  TimeStepping time_stepping = TS_EULER;

  bool multiple_outputs = false;

  // Parallel stuff
  ParallelRange range_tot;
  ParallelRange range_dom;
  ParallelRange range_xbound;
  ParallelRange range_ybound;
  ParallelRange range_slopes;

  // Run
  std::string problem;

  // All the physics
  DeviceParams device_params;

  // Misc 
  int seed;
  int log_frequency;
  
  struct value_container {
    std::string value;
    bool from_file = false;
    bool used = false;
  };
  std::map<std::string, std::map<std::string, value_container>> _values;
  
  template<typename T>
  void registerValue(std::string section, std::string name, const T& default_value) {
    // TODO: revoir la logique car affiche unused et default à chaque paramètre.
    // Les valeurs sont par contre correctes.
    auto hasValue = [&](const std::string& section, const std::string& name) {
      return (this->_values.count(section) != 0) && (this->_values.at(section).count(name) != 0);
    };

    bool is_present_in_file = hasValue(section, name);
    if (is_present_in_file) {
      this->_values[section][name].used = true;
      this->_values[section][name].from_file = true;
    }
    if constexpr (std::is_same_v<T, std::string>){
      this->_values[section][name].value = default_value;
    }
    else {
      this->_values[section][name].value = std::to_string(default_value);
    }
  }
  bool GetBoolean(std::string section, std::string name, bool default_value){
    bool res = this->reader.GetBoolean(section, name, default_value);
    registerValue(section, name, res);
    return res;
  }
  
  int GetInteger(std::string section, std::string name, int default_value){
    int res = this->reader.GetInteger(section, name, default_value);
    registerValue(section, name, res);
    return res;
  }
  
  real_t GetFloat(std::string section, std::string name, real_t default_value){
    real_t res = this->reader.GetFloat(section, name, default_value);
    registerValue(section, name, res);
    return res;
  }
  std::string Get(std::string section, std::string name, std::string default_value){
    std::string res = this->reader.Get(section, name, default_value);
    registerValue(section, name, res);
    return res;
  }

  void outputValues(std::ostream& o){
    constexpr std::string::size_type name_width = 20;
    constexpr std::string::size_type value_width = 20;
    auto initial_format = o.flags();
    std::string problem = this->Get("physics", "problem", "unknown");
    o << "Parameters used for the problem: " << problem << std::endl;
    o << std::left;
    for( auto p_section : this->_values )
    {
      const std::string& section_name = p_section.first;
      const std::map<std::string, value_container>& map_section = p_section.second;

      o << "\n[" << section_name << "]" << std::endl;
      for( auto p_var : map_section )
      {
        const std::string& var_name = p_var.first;
        const value_container& val = p_var.second;

        o << std::setw(std::max(var_name.length(),name_width)) << var_name << " = " << std::setw(std::max(val.value.length(), value_width)) << val.value << std::endl;
      }
    }
    o.flags(initial_format);
  }
};


// Helper to get the position in the mesh
KOKKOS_INLINE_FUNCTION
Pos getPos(const DeviceParams& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

Params readInifile(std::string filename) {
  // Params reader(filename);
  Params res;
  res.reader = INIReader(filename);
  // Run
  res.tend = res.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = res.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = res.Get("run", "restart_file", "");
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
    
  res.save_freq = res.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = res.Get("run", "output_filename", "run");
  res.path_out = res.Get("run", "output_path", ".");

  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = read_map(res.reader, ts_map, "solvers", "time_stepping", "euler");
  res.problem = res.Get("physics", "problem", "blast");


  // Misc
  res.seed = res.GetInteger("misc", "seed", 12345);
  res.log_frequency = res.GetInteger("misc", "log_frequency", 10);

  // All device parameters
  res.device_params.init_from_inifile(res.reader);

  // Parallel ranges
  res.range_tot = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Nty});
  res.range_dom = ParallelRange({res.device_params.ibeg, res.device_params.jbeg}, {res.device_params.iend, res.device_params.jend});
  res.range_xbound = ParallelRange({0, res.device_params.jbeg}, {res.device_params.Ng, res.device_params.jend});
  res.range_ybound = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Ng});
  res.range_slopes = ParallelRange({res.device_params.ibeg-1, res.device_params.jbeg-1}, {res.device_params.iend+1, res.device_params.jend+1});
  
  return res;
} 
}


// All states operations
#include "States.h"

namespace fv2d {
void consToPrim(Array U, Array Q, const Params &full_params) {
  auto params = full_params.device_params;
  Kokkos::parallel_for( "Conservative to Primitive", 
                        full_params.range_tot,
                        KOKKOS_LAMBDA(const int i, const int j) {
                          State Uloc = getStateFromArray(U, i, j);
                          State Qloc = consToPrim(Uloc, params);
                          setStateInArray(Q, i, j, Qloc);
                        });
}

void primToCons(Array &Q, Array &U, const Params &full_params) {
  auto params = full_params.device_params;
  Kokkos::parallel_for( "Primitive to Conservative", 
                        full_params.range_tot,
                        KOKKOS_LAMBDA(const int i, const int j) {
                          State Qloc = getStateFromArray(Q, i, j);
                          State Uloc = primToCons(Qloc, params);
                          setStateInArray(U, i, j, Uloc);
                        });
}

void checkNegatives(Array &Q, const Params &full_params) {
  uint64_t negative_density  = 0;
  uint64_t negative_pressure = 0;
  uint64_t nan_count = 0;

  Kokkos::parallel_reduce(
    "Check negative density/pressure", 
    full_params.range_dom,
    KOKKOS_LAMBDA(const int i, const int j, uint64_t& lnegative_density, uint64_t& lnegative_pressure, uint64_t& lnan_count) {
      constexpr real_t eps = 1.0e-6;
      if (Q(j, i, IR) < 0) {
        Q(j, i, IR) = eps;
        lnegative_density++;
      }
      if (Q(j, i, IP) < 0) {
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


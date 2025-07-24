#pragma once

#include <cmath>
#include <string>
#include <iomanip>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>

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

#ifdef MHD
constexpr int Nfields = 9;
#else
constexpr int Nfields = 4;
#endif

using Pos   = Kokkos::Array<real_t, 2>;
using Vect  = Kokkos::Array<real_t, 3>;
using State = Kokkos::Array<real_t, Nfields>;
using Array = Kokkos::View<real_t***>;
using ParallelRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using Matrix = Kokkos::Array<Kokkos::Array<real_t, Nfields>, Nfields>;

struct RestartInfo {
  real_t time;
  int iteration;
};

enum IDir : uint8_t {
  IX = 0,
  IY = 1,
  IZ = 2
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
  IBZ = 7,
  IPSI = 8
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
  HLLD,
  FIVEWAVES,
  IDEALGLM
};

enum DivCleaning {
  NO_DC,
  DEDNER, // hyperbolic div-cleaning
  DERIGS // entropy consistent
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

enum GravityMode {
  GRAV_NONE,
  GRAV_CONSTANT,
  GRAV_ANALYTICAL
};

enum AnalyticalGravityMode {
  AGM_HOT_BUBBLE
};

// All parameters that should be copied on the device
struct DeviceParams { 
  // Thermodynamics
  real_t gamma0 = 5.0/3.0;
  
  // Gravity
  GravityMode gravity_mode;
  real_t gx, gy;
  AnalyticalGravityMode analytical_gravity_mode;
  bool well_balanced_flux_at_y_bc = false;
  bool well_balanced = false;
  real_t smallr = 1.0e-10;
  real_t smallp = 1.0e-10;
  // Thermal conduction
  
  // Thermal conductivity
  bool thermal_conductivity_active;
  ThermalConductivityMode thermal_conductivity_mode;
  real_t kappa;
  BCTC_Mode bctc_ymin, bctc_ymax;
  real_t bctc_ymin_value, bctc_ymax_value;
  
  // Heating
  bool heating_active;
  HeatingMode heating_mode;
  bool log_total_heating;

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
  real_t iso3_bx, iso3_by;
  // Hot bubble
  real_t hot_bubble_g0;

  // Kelvin-Helmholtz
  real_t kh_y1, kh_y2;
  real_t kh_a;
  real_t kh_sigma;
  real_t kh_rho_fac;
  real_t kh_uflow;
  real_t kh_amp;
  real_t kh_P0;
  
  // Boundaries
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  
  // Godunov
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  real_t CFL = 0.1;
  
  // Divergence Cleaning - MHD only
  DivCleaning div_cleaning = NO_DC;
  real_t cr = 0.18; // GLMMHD
  real_t GLM_scale = 1.0; // \in ]0;1] - IdealGLM scale factor for value of cleaning speed ch
  
  
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
  // Misc stuff
  real_t epsilon = 1.0e-6;
  
  void init_from_inifile(INIReader &reader) {
    
    
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
      {"reflecting",         BC_REFLECTING},
      {"absorbing",          BC_ABSORBING},
      {"periodic",           BC_PERIODIC},
      {"tri_layer_damping",  BC_TRILAYER_DAMPING}
    };
    boundary_x = read_map(reader, bc_map, "run", "boundaries_x", "reflecting");
    boundary_y = read_map(reader, bc_map, "run", "boundaries_y", "reflecting");
    std::map<std::string, ReconstructionType> recons_map{
      {"pcm",    PCM},
      {"pcm_wb", PCM_WB},
      {"pcm_wb2", PCM_WB2},
      {"plm",    PLM}
    };
    reconstruction = read_map(reader, recons_map, "solvers", "reconstruction", "pcm");
    
    std::map<std::string, RiemannSolver> riemann_map{
      {"hll", HLL},
      {"hllc", HLLC},
      {"hlld", HLLD},
      {"fivewaves", FIVEWAVES},
      {"idealGLM", IDEALGLM}
    };
    riemann_solver = read_map(reader, riemann_map, "solvers", "riemann_solver", "hllc");
    std::map<std::string, DivCleaning> div_cleaning_map{
      {"none", NO_DC},
      {"dedner", DEDNER},
      {"derigs", DERIGS}
    };
    div_cleaning = read_map(reader, div_cleaning_map, "solvers", "div_cleaning", "dedner");
    if (div_cleaning == DERIGS) {
      throw std::runtime_error("Derigs div cleaning is not implemented yet !");
    };
    // Physics
    epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
    gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
    m1      = reader.GetFloat("polytrope", "m1", 1.0);
    theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
    m2      = reader.GetFloat("polytrope", "m2", 1.0);
    theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
    well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);
    cr      = reader.GetFloat("physics", "cr", 0.18);
    GLM_scale = reader.GetFloat("physics", "GLM_scale", 1.0);
    

    // Gravity
    std::map<std::string, GravityMode> gravity_map{
      {"none",       GRAV_NONE},
      {"constant",   GRAV_CONSTANT},
      {"analytical", GRAV_ANALYTICAL}
    };
    gravity_mode = read_map(reader, gravity_map, "gravity", "mode", "none");

    gx = reader.GetFloat("gravity", "gx", 0.0);
    gy = reader.GetFloat("gravity", "gy", 0.0);

    std::map<std::string, AnalyticalGravityMode> analytical_gravity_map{
      {"hot_bubble", AGM_HOT_BUBBLE}
    };
    analytical_gravity_mode = read_map(reader, analytical_gravity_map, "gravity", "analytical_mode", "hot_bubble");

    // Thermal conductivity
    thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
    std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
      {"constant", TCM_CONSTANT},
      {"B02",      TCM_B02},
      {"tri_layer", TCM_C2020_TRI},
      {"iso-three", TCM_ISO3},
    };
    thermal_conductivity_mode = read_map(reader, thermal_conductivity_map, "thermal_conduction", "conductivity_mode", "constant");
    kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);
    std::map<std::string, BCTC_Mode> bctc_map{
      {"none",              BCTC_NONE},
      {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
      {"fixed_gradient",    BCTC_FIXED_GRADIENT},
      {"no_flux",           BCTC_NO_FLUX},
      {"no_conduction",     BCTC_NO_CONDUCTION},
    };
    bctc_ymin = read_map(reader, bctc_map, "thermal_conduction", "bc_ymin", "none");
    bctc_ymax = read_map(reader, bctc_map, "thermal_conduction", "bc_ymax", "none");
    bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_ymin_value", 1.0);
    bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_ymax_value", 1.0);

    // Viscosity
    viscosity_active = reader.GetBoolean("viscosity", "active", false);
    std::map<std::string, ViscosityMode> viscosity_map{
      {"constant", VSC_CONSTANT},
    };
    viscosity_mode = read_map(reader, viscosity_map, "viscosity", "viscosity_mode", "constant");
    mu = reader.GetFloat("viscosity", "mu", 0.0);

    // Heating function 
    heating_active = reader.GetBoolean("heating", "active", false);
    std::map<std::string, HeatingMode> heating_map{
      {"C2020", HM_C2020},
      {"tri_layer", HM_C2020_TRI},
      {"isothermal_cooling", HM_COOLING_ISO}
    };
    heating_mode = read_map(reader, heating_map, "heating", "mode", "tri_layer");
    log_total_heating = reader.GetBoolean("misc", "log_total_heating", false);

    // H84
    h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

    // C91
    c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

    // C20
    c20_H = reader.GetFloat("C20", "H", 0.2);
    c20_heating_fac = reader.GetFloat("C20", "heating_fac", 2.0);

    // Tri-layer
    tri_y1   = reader.GetFloat("tri_layer", "y1", 1.0);
    tri_y2   = reader.GetFloat("tri_layer", "y2", 2.0);
    tri_pert = reader.GetFloat("tri_layer", "perturbation", 1.0e-3);
    tri_k1   = reader.GetFloat("tri_layer", "kappa1", 0.07);
    tri_k2   = reader.GetFloat("tri_layer", "kappa2", 1.5);
    T0       = reader.GetFloat("tri_layer", "T0", 1.0);
    rho0     = reader.GetFloat("tri_layer", "rho0", 1.0);
    
    // Isothermal triple layer
    iso3_dy0    = reader.GetFloat("iso_three_layer", "dy0", 1.0);
    iso3_dy1    = reader.GetFloat("iso_three_layer", "dy1", 2.0);
    iso3_dy2    = reader.GetFloat("iso_three_layer", "dy2", 2.0);
    iso3_theta1 = reader.GetFloat("iso_three_layer", "theta1", 2.0);
    iso3_theta2 = reader.GetFloat("iso_three_layer", "theta2", 2.0);
    iso3_pert   = reader.GetFloat("iso_three_layer", "perturbation", 1.0e-3);
    iso3_k1     = reader.GetFloat("iso_three_layer", "k1", 0.07);
    iso3_k2     = reader.GetFloat("iso_three_layer", "k2", 1.5);
    iso3_m1     = reader.GetFloat("iso_three_layer", "m1", 1.0);
    iso3_m2     = reader.GetFloat("iso_three_layer", "m2", 1.0);
    iso3_T0     = reader.GetFloat("iso_three_layer", "T0", 1.0);
    iso3_rho0   = reader.GetFloat("iso_three_layer", "rho0", 1.0);
    iso3_bx      = reader.GetFloat("iso_three_layer", "bx", 0.0);
    iso3_by      = reader.GetFloat("iso_three_layer", "by", 0.0);
    // Hot bubble
    hot_bubble_g0 = reader.GetFloat("hot_bubble", "g0", 0.0);

    // Kelvin-Helmholtz
    kh_a = reader.GetFloat("kelvin_helmholtz", "a", 0.05);
    kh_amp = reader.GetFloat("kelvin_helmholtz", "amp", 0.01);
    kh_P0  = reader.GetFloat("kelvin_helmholtz", "P0", 1.0);
    kh_rho_fac = reader.GetFloat("kelvin_helmholtz", "rho_fac", 0.0);
    kh_sigma = reader.GetFloat("kelvin_helmholtz", "sigma", 0.2);
    kh_uflow = reader.GetFloat("kelvin_helmholts", "uflow", 1.0);
    kh_y1 = reader.GetFloat("kelvin_helmholts", "y1", 0.5);
    kh_y2 = reader.GetFloat("kelvin_helmholts", "y2", 1.5);
  }
};

// All the parameters
struct Params {
  real_t save_freq;
  real_t tend;
  INIReader reader;
  std::string filename_out = "run";
  std::string restart_file = "";
  bool restart_mhd_from_hydro = false;
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
  std::string inter_problem; // function to re-initialize a md run from hydro
  // All the physics
  DeviceParams device_params;

  // Misc 
  int seed;
  int log_frequency;
  bool log_energy_contributions;
  int log_energy_frequency;
  
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

// Printing every single parameter of the run
void print_ini_file(const Params &p) {
  std::cout << std::endl << std::endl;
  std::cout << "===========================================" << std::endl;
  std::cout << "== Parameters read : " << std::endl;
  std::cout << std::endl << "[HOST]" << std::endl;
  std::cout << "Problem          = " << p.problem << std::endl;
  std::cout << "Filename out     = " << p.filename_out << std::endl;
  std::cout << "T end            = " << p.tend << std::endl;
  std::cout << "Save frequency   = " << p.save_freq << std::endl;
  std::cout << "Restart file     = " << p.restart_file << std::endl;
  std::cout << "Multiple outputs = " << p.multiple_outputs << std::endl;
  std::cout << "Time stepping    = " << p.time_stepping << std::endl;
  
  std::cout << std::endl << "[DEVICE]" << std::endl;
  auto &dp = p.device_params;
  std::cout << " -- Grid -- " << std::endl;
  std::cout << "Nx                 = " << dp.Nx << std::endl;
  std::cout << "Ny                 = " << dp.Ny << std::endl;
  std::cout << "Nghosts            = " << dp.Ng << std::endl;
  std::cout << "xmin               = " << dp.xmin << std::endl;
  std::cout << "xmax               = " << dp.xmax << std::endl;
  std::cout << "ymin               = " << dp.ymin << std::endl;
  std::cout << "ymax               = " << dp.ymax << std::endl;
  std::cout << "Ntx                = " << dp.Ntx << std::endl;
  std::cout << "Nty                = " << dp.Nty << std::endl;
  std::cout << "ibeg, iend         = " << dp.ibeg << " " << dp.iend << std::endl;
  std::cout << "jbeg, jend         = " << dp.jbeg << " " << dp.jend << std::endl;
  std::cout << "dx, dy             = " << dp.dx << " " << dp.dy << std::endl;
  std::cout << "boundary X         = " << dp.boundary_x << std::endl;
  std::cout << "boundary Y         = " << dp.boundary_y << std::endl;
  
  std::cout << std::endl << " -- Numerics -- " << std::endl;
  std::cout << "reconstruction     = " << dp.reconstruction << std::endl;
  std::cout << "riemann_solver     = " << dp.riemann_solver << std::endl;
  std::cout << "CFL                = " << dp.CFL << std::endl;
  std::cout << "epsilon            = " << dp.epsilon << std::endl;

  std::cout << std::endl << " -- Physics -- " << std::endl;
  std::cout << "gamma0             = " << dp.gamma0 << std::endl;
  std::cout << "gravity active     = " << dp.gravity_mode << std::endl;
  std::cout << "gx                 = " << dp.gx << std::endl;
  std::cout << "gy                 = " << dp.gy << std::endl;
  std::cout << "m1, m2             = " << dp.m1 << ", " << dp.m2 << std::endl;
  std::cout << "theta1, theta2     = " << dp.theta1 << ", " << dp.theta2 << std::endl;
  std::cout << "wb flux at y bc    = " << dp.well_balanced_flux_at_y_bc << std::endl;
  std::cout << "thermal conduction = " << dp.thermal_conductivity_active << std::endl;
  std::cout << "TC mode            = " << dp.thermal_conductivity_mode << std::endl;
  std::cout << "kappa              = " << dp.kappa << std::endl;
  std::cout << "bctc_ymin          = " << dp.bctc_ymin << std::endl; 
  std::cout << "bctc_ymax          = " << dp.bctc_ymax << std::endl;
  std::cout << "bctc_ymin_value    = " << dp.bctc_ymin_value << std::endl; 
  std::cout << "bctc_ymax_value    = " << dp.bctc_ymax_value << std::endl;
  std::cout << "viscosity          = " << dp.viscosity_active << std::endl;
  std::cout << "mu                 = " << dp.mu << std::endl;
  std::cout << "viscosity_mode     = " << dp.viscosity_mode << std::endl;
  std::cout << "heating            = " << dp.heating_active << std::endl;
  std::cout << "heating_mode       = " << dp.heating_mode << std::endl;
  std::cout << "iso3_dy0           = " << dp.iso3_dy0 << std::endl;
  std::cout << "iso3_dy1           = " << dp.iso3_dy1 << std::endl;
  std::cout << "iso3_dy2           = " << dp.iso3_dy2 << std::endl;
  std::cout << "iso3_theta1        = " << dp.iso3_theta1 << std::endl;
  std::cout << "iso3_theta2        = " << dp.iso3_theta2 << std::endl;
  std::cout << "iso3_m1            = " << dp.iso3_m1 << std::endl;
  std::cout << "iso3_m2            = " << dp.iso3_m2 << std::endl;
  std::cout << "iso3_k1            = " << dp.iso3_k1 << std::endl;
  std::cout << "iso3_k2            = " << dp.iso3_k2 << std::endl;
  std::cout << "iso3_T0            = " << dp.iso3_T0 << std::endl;
  std::cout << "iso3_rho0          = " << dp.iso3_rho0 << std::endl;
}

Params readInifile(std::string filename) {
  // Params reader(filename);
  Params res;
  res.reader = INIReader(filename);
  // Run
  res.tend = res.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = res.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = res.Get("run", "restart_file", "");
  res.restart_mhd_from_hydro = res.GetBoolean("run", "restart_mhd_from_hydro", false);
  
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
    
  res.save_freq = res.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = res.Get("run", "output_filename", "run");

  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = read_map(res.reader, ts_map, "solvers", "time_stepping", "euler");
  res.problem = res.Get("physics", "problem", "blast");

  // Misc
  res.seed = res.GetInteger("misc", "seed", 12345);
  res.log_frequency = res.GetInteger("misc", "log_frequency", 10);
  res.log_energy_contributions = res.GetBoolean("misc", "log_energy_contributions", false);
  res.log_energy_frequency = res.GetFloat("misc", "log_energy_frequency", 10);

  // All device parameters
  res.device_params.init_from_inifile(res.reader);

  // Parallel ranges
  res.range_tot = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Nty});
  res.range_dom = ParallelRange({res.device_params.ibeg, res.device_params.jbeg}, {res.device_params.iend, res.device_params.jend});
  res.range_xbound = ParallelRange({0, res.device_params.jbeg}, {res.device_params.Ng, res.device_params.jend});
  res.range_ybound = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Ng});
  res.range_slopes = ParallelRange({res.device_params.ibeg-1, res.device_params.jbeg-1}, {res.device_params.iend+1, res.device_params.jend+1});

  print_ini_file(res);

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

      for (int ivar=0; ivar < NfieldsM; ++ivar)
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


} // namespace fv2d

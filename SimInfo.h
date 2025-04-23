#pragma once

#include <cmath>
#include <string>
#include <iomanip>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>

namespace fv2d {

using real_t = double;

#ifdef MHD
constexpr int Nfields = 9;
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
  FIVEWAVES
};

enum DivCleaning {
  NO_DC,
  DEDNER, // hyperbolic div-cleaning
  DERIGS // entropy consistent
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
struct Params{
  INIReader reader;
  real_t save_freq;
  real_t tend;
  std::string filename_out = "run";
  std::string restart_file = "";
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  DivCleaning div_cleaning = DEDNER;
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
  real_t cr = 0.1; // GLMMHD

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
Pos getPos(const Params& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

Params readInifile(std::string filename) {
  // Params reader(filename);
  Params res;
  res.reader = INIReader(filename);
  // INIReader& reader = res.reader;
  // Mesh
  res.Nx = res.GetInteger("mesh", "Nx", 32);
  res.Ny = res.GetInteger("mesh", "Ny", 32);
  res.Ng = res.GetInteger("mesh", "Nghosts", 2);
  res.xmin = res.GetFloat("mesh", "xmin", 0.0);
  res.xmax = res.GetFloat("mesh", "xmax", 1.0);
  res.ymin = res.GetFloat("mesh", "ymin", 0.0);
  res.ymax = res.GetFloat("mesh", "ymax", 1.0);

  res.Ntx  = res.Nx + 2*res.Ng;
  res.Nty  = res.Ny + 2*res.Ng;
  res.ibeg = res.Ng;
  res.iend = res.Ng+res.Nx;
  res.jbeg = res.Ng;
  res.jend = res.Ng+res.Ny;

  res.dx = (res.xmax-res.xmin) / res.Nx;
  res.dy = (res.ymax-res.ymin) / res.Ny;

  // Run
  res.tend = res.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = res.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = res.Get("run", "restart_file", "");
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
    
  res.save_freq = res.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = res.Get("run", "output_filename", "run");

  std::string tmp;
  tmp = res.Get("run", "boundaries_x", "reflecting");
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",         BC_REFLECTING},
    {"absorbing",          BC_ABSORBING},
    {"periodic",           BC_PERIODIC}
  };
  res.boundary_x = bc_map[tmp];
  tmp = res.Get("run", "boundaries_y", "reflecting");
  res.boundary_y = bc_map[tmp];

  tmp = res.Get("solvers", "reconstruction", "pcm");
  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",    PCM},
    {"pcm_wb", PCM_WB},
    {"plm",    PLM}
  };
  res.reconstruction = recons_map[tmp];

  tmp = res.Get("solvers", "riemann_solver", "hllc");
  std::map<std::string, RiemannSolver> riemann_map{
    {"hll", HLL},
    {"hllc", HLLC},
    {"hlld", HLLD},
    {"fivewaves", FIVEWAVES}
  };
  res.riemann_solver = riemann_map[tmp];
  tmp = res.Get("solvers", "div_cleaning", "dedner");
  std::map<std::string, DivCleaning> div_cleaning_map{
    {"none", NO_DC},
    {"dedner", DEDNER},
    {"derigs", DERIGS}
  };
  res.div_cleaning = div_cleaning_map[tmp];
  if (res.div_cleaning == DERIGS) {
    throw std::runtime_error("Derigs div cleaning is not implemented yet !");
  };
  tmp = res.Get("solvers", "time_stepping", "euler");
  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = ts_map[tmp];

  res.CFL = res.GetFloat("solvers", "CFL", 0.8);

  // Physics
  res.epsilon = res.GetFloat("misc", "epsilon", 1.0e-6);
  res.gamma0  = res.GetFloat("physics", "gamma0", 5.0/3.0);
  res.gravity = res.GetBoolean("physics", "gravity", false);
  res.g       = res.GetFloat("physics", "g", 0.0);
  res.cr      = res.GetFloat("physics", "cr", 0.1);
  res.m1      = res.GetFloat("polytrope", "m1", 1.0);
  res.theta1  = res.GetFloat("polytrope", "theta1", 10.0);
  res.m2      = res.GetFloat("polytrope", "m2", 1.0);
  res.theta2  = res.GetFloat("polytrope", "theta2", 10.0);
  res.problem = res.Get("physics", "problem", "blast");
  res.well_balanced_flux_at_y_bc = res.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);

  // Thermal conductivity
  res.thermal_conductivity_active = res.GetBoolean("thermal_conduction", "active", false);
  tmp = res.Get("thermal_conduction", "conductivity_mode", "constant");
  std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
    {"constant", TCM_CONSTANT},
    {"B02",      TCM_B02}
  };
  res.thermal_conductivity_mode = thermal_conductivity_map[tmp];
  res.kappa = res.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT}
  };
  tmp = res.Get("thermal_conduction", "bc_xmin", "none");
  res.bctc_ymin = bctc_map[tmp];
  tmp = res.Get("thermal_conduction", "bc_xmax", "none");
  res.bctc_ymax = bctc_map[tmp];
  res.bctc_ymin_value = res.GetFloat("thermal_conduction", "bc_xmin_value", 1.0);
  res.bctc_ymax_value = res.GetFloat("thermal_conduction", "bc_xmax_value", 1.0);

  // Viscosity
  res.viscosity_active = res.GetBoolean("viscosity", "active", false);
  tmp = res.Get("viscosity", "viscosity_mode", "constant");
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  res.viscosity_mode = viscosity_map[tmp];
  res.mu = res.GetFloat("viscosity", "mu", 0.0);

  // H84
  res.h84_pert = res.GetFloat("H84", "perturbation", 1.0e-4);

  // C91
  res.c91_pert = res.GetFloat("C91", "perturbation", 1.0e-3);

  // Misc
  res.seed = res.GetInteger("misc", "seed", 12345);
  res.log_frequency = res.GetInteger("misc", "log_frequency", 10);


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

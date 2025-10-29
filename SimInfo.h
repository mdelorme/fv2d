#pragma once

#include <cmath>
#include <string>
#include <iomanip>
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

struct RestartInfo {
  real_t time;
  int iteration;
};

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
  HLLC,
  FSLP,
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

enum GravityMode {
  GRAV_NONE,
  GRAV_CONSTANT,
  GRAV_ANALYTICAL
};

enum AnalyticalGravityMode {
  AGM_HOT_BUBBLE
};

// Add functions HasSection and HasValue to INIReader, remove this when jtilly/inih.git will be updated
struct IniReader : INIReader {
  using INIReader::INIReader, INIReader::GetBoolean, INIReader::GetInteger, INIReader::GetFloat, INIReader::Get;
  using INIReader::_values, INIReader::_sections;

  bool HasSection(const std::string& section) const
  {
      const std::string key = MakeKey(section, "");
      std::map<std::string, std::string>::const_iterator pos = _values.lower_bound(key);
      if (pos == _values.end())
          return false;
      // Does the key at the lower_bound pos start with "section"?
      return pos->first.compare(0, key.length(), key) == 0;
  }

  bool HasValue(const std::string& section, const std::string& name) const
  {
      std::string key = MakeKey(section, name);
      return _values.count(key);
  }
};

// Reader 
struct Reader {
  Reader() = default;
  Reader(const std::string &filename) 
  : reader(filename) {};
  ~Reader() = default;

  struct value_container {
    std::string value;
    bool from_file = false;
    bool is_default_value = true;
  };
  std::map<std::string, std::map<std::string, value_container>> _values;
  IniReader reader;

  template<typename T>
  void registerValue(std::string section, std::string name, const T& value, bool is_default_value) {
    
    std::transform(section.begin(), section.end(), section.begin(), ::tolower);
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    auto isAlreadyPresent = [&](const std::string& section, const std::string& name) {
      return (this->_values.count(section) != 0) && (this->_values.at(section).count(name) != 0);
    };
    auto isPresent = [&](const std::string& section, const std::string& name) {
      return (this->reader.HasSection(section) && this->reader.HasValue(section, name));
    };

    bool is_already_present_in_file = isAlreadyPresent(section, name);
    if (is_already_present_in_file) {
      throw std::runtime_error(std::string("parameter already set : ") + name);
    }
    bool is_present_in_file = isPresent(section, name);
    if (is_present_in_file) {
      this->_values[section][name].from_file = true;
      this->_values[section][name].is_default_value = is_default_value;
    }

    if constexpr (std::is_same_v<T, std::string>){
      this->_values[section][name].value = value;
    }
    else if constexpr (std::is_same_v<T, bool>) {
      this->_values[section][name].value = (value) ? "true" : "false";
    }
    else if constexpr (std::is_floating_point_v<T>) {
      std::ostringstream os; os << std::scientific << std::setprecision(12) << value;
      this->_values[section][name].value = os.str();
    }
    else {
      this->_values[section][name].value = std::to_string(value);
    }
  }
  bool GetBoolean(std::string section, std::string name, bool default_value){
    bool res = this->reader.GetBoolean(section, name, default_value); 
    registerValue(section, name, res, res == default_value);
    return res;
  }
  
  int GetInteger(std::string section, std::string name, int default_value){
    int res = this->reader.GetInteger(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  
  real_t GetFloat(std::string section, std::string name, real_t default_value){
    real_t res = this->reader.GetFloat(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  std::string Get(std::string section, std::string name, std::string default_value){
    std::string res = this->reader.Get(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  template<typename T>
  auto GetMapValue(const std::map<std::string, T>& map, const std::string& section, const std::string& name, const std::string& default_value){
    std::string tmp;
    tmp = this->Get(section, name, default_value);

    if (map.count(tmp) == 0) {
      tmp = "\nallowed values: ";
      for (auto elem : map) tmp += elem.first + ", ";
      throw std::runtime_error(std::string("bad parameter for ") + name + ": " + tmp);
    }
    return map.at(tmp);
  };

  void outputValues(std::ostream& o){
    constexpr std::string::size_type name_width = 26;
    constexpr std::string::size_type value_width = 20;
    auto initial_format = o.flags();
    std::string problem = this->_values["physics"]["problem"].value;
    o << "; Parameters used for the problem: " << problem << std::endl;
    o << std::left;
    
    for( auto p_section : this->_values )
    {
      const std::string& section_name = p_section.first;
      const std::map<std::string, value_container>& map_section = p_section.second;

      // skip section if it doesn't appear in the .ini
      if ( !this->reader.HasSection(p_section.first) )
        continue;

      // skip section if there is only default values
      /*
      for( auto p_var : map_section ) 
        if ( p_var.second.from_file ) // or p_var.second.is_default_value
          break;
      */
    
      o << "\n[" << section_name << "]" << std::endl;
      
      for( auto p_var : map_section )
      {
        const std::string& var_name = p_var.first;
        const value_container& val = p_var.second;

        o << std::setw(std::max(var_name.length(),name_width)) << var_name 
          << " = " << std::setw(std::max(val.value.length(), value_width)) << val.value 
          << (val.from_file ? "" : " ; default ")
          << std::endl;
      }
    }
    o.flags(initial_format);
  }
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
  real_t fslp_K;
  
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

  // Gresho vortex
  real_t gresho_density, gresho_Mach;
  
  // Boundaries
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  
  // Godunov
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

  // Misc stuff
  real_t epsilon = 1.0e-6;
  
  void init_from_inifile(Reader &reader) {
    
    
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
      {"periodic",           BC_PERIODIC}
    };
    boundary_x = reader.GetMapValue(bc_map, "run", "boundaries_x", "reflecting");
    boundary_y = reader.GetMapValue(bc_map, "run", "boundaries_y", "reflecting");

    std::map<std::string, ReconstructionType> recons_map{
      {"pcm",    PCM},
      {"pcm_wb", PCM_WB},
      {"plm",    PLM}
    };
    reconstruction = reader.GetMapValue(recons_map, "solvers", "reconstruction", "pcm");

    std::map<std::string, RiemannSolver> riemann_map{
      {"hll", HLL},
      {"hllc", HLLC},
      {"fslp", FSLP}
    };
    riemann_solver = reader.GetMapValue(riemann_map, "solvers", "riemann_solver", "hllc");

    // Physics
    epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
    gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
    m1      = reader.GetFloat("polytrope", "m1", 1.0);
    theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
    m2      = reader.GetFloat("polytrope", "m2", 1.0);
    theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
    well_balanced_flux_at_y_bc = reader.GetBoolean("physics", "well_balanced_flux_at_y_bc", false);
    fslp_K  = reader.GetFloat("physics", "fslp_K", 1.1);

    // Gravity
    std::map<std::string, GravityMode> gravity_map{
      {"none",       GRAV_NONE},
      {"constant",   GRAV_CONSTANT},
      {"analytical", GRAV_ANALYTICAL}
    };
    gravity_mode = reader.GetMapValue(gravity_map, "gravity", "mode", "none");

    gx = reader.GetFloat("gravity", "gx", 0.0);
    gy = reader.GetFloat("gravity", "gy", 0.0);

    std::map<std::string, AnalyticalGravityMode> analytical_gravity_map{
      {"hot_bubble", AGM_HOT_BUBBLE}
    };
    analytical_gravity_mode = reader.GetMapValue(analytical_gravity_map, "gravity", "analytical_mode", "hot_bubble");

    // Thermal conductivity
    thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
    std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
      {"constant", TCM_CONSTANT},
      {"B02",      TCM_B02}
    };
    thermal_conductivity_mode = reader.GetMapValue(thermal_conductivity_map, "thermal_conduction", "conductivity_mode", "constant");
    kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

    std::map<std::string, BCTC_Mode> bctc_map{
      {"none",              BCTC_NONE},
      {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
      {"fixed_gradient",    BCTC_FIXED_GRADIENT}
    };
    bctc_ymin = reader.GetMapValue(bctc_map, "thermal_conduction", "bc_ymin", "none");
    bctc_ymax = reader.GetMapValue(bctc_map, "thermal_conduction", "bc_ymax", "none");
    bctc_ymin_value = reader.GetFloat("thermal_conduction", "bc_ymin_value", 1.0);
    bctc_ymax_value = reader.GetFloat("thermal_conduction", "bc_ymax_value", 1.0);

    // Viscosity
    viscosity_active = reader.GetBoolean("viscosity", "active", false);
    std::map<std::string, ViscosityMode> viscosity_map{
      {"constant", VSC_CONSTANT},
    };
    viscosity_mode = reader.GetMapValue(viscosity_map, "viscosity", "viscosity_mode", "constant");
    mu = reader.GetFloat("viscosity", "mu", 0.0);

    // H84
    h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

    // C91
    c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

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
  
    // Gresho vortex
    gresho_density  = reader.GetFloat("gresho_vortex", "density",  1.0);
    gresho_Mach     = reader.GetFloat("gresho_vortex", "Mach",     0.1);
  }
};

// All the parameters
struct Params {
  real_t save_freq;
  real_t tend;
  Reader reader;
  std::string filename_out = "run";
  std::string output_path = "./";
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
  real_t epsilon_reset_negative; // fixed value if negative value is encountered 
};


// Helper to get the position in the mesh
KOKKOS_INLINE_FUNCTION
Pos getPos(const DeviceParams& params, int i, int j) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy};
}

void checkValidityIni(Params &params) {
  auto &ini_sections = params.reader.reader._sections;
  auto &ini_keyvalues = params.reader.reader._values; // format: { "section=key", "value" }
  auto &valid_keyvalues = params.reader._values;      // format: { "key", struct value }

  for (auto s : ini_sections) {
    bool section_ok = valid_keyvalues.count(s) > 0;
    if (!section_ok) {
      std::cerr << "WARNING: section [" << s << "] is unknown." << std::endl;
      continue;
    }
    
    for (auto [k,v] : ini_keyvalues) {
      if(k.starts_with(s + "=")) {
        auto value = k.substr(s.length()+1);
        bool value_ok = valid_keyvalues[s].count(value) > 0;
        if (!value_ok)
        std::cerr << "WARNING: parameter `" << value << "` in section [" << s << "] is unknown." << std::endl;
      }
    }
  }
}

Params readInifile(std::string filename) {
  Params res;
  res.reader = Reader(filename);
  auto &reader = res.reader;
  
  // Run
  res.tend = reader.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = reader.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = reader.Get("run", "restart_file", "");
  
  res.save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = reader.Get("run", "output_filename", "run");
  res.output_path = reader.Get("run", "output_path", "./");

  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = reader.GetMapValue(ts_map, "solvers", "time_stepping", "euler");
  res.problem = reader.Get("physics", "problem", "blast");

  // Misc
  res.seed = reader.GetInteger("misc", "seed", 12345);
  res.log_frequency = reader.GetInteger("misc", "log_frequency", 10);
  res.epsilon_reset_negative = reader.GetFloat("misc", "epsilon_reset_negative", 1.0e-8);

  // All device parameters
  res.device_params.init_from_inifile(res.reader);
  
  // Parallel ranges
  res.range_tot = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Nty});
  res.range_dom = ParallelRange({res.device_params.ibeg, res.device_params.jbeg}, {res.device_params.iend, res.device_params.jend});
  res.range_xbound = ParallelRange({0, res.device_params.jbeg}, {res.device_params.Ng, res.device_params.jend});
  res.range_ybound = ParallelRange({0, 0}, {res.device_params.Ntx, res.device_params.Ng});
  res.range_slopes = ParallelRange({res.device_params.ibeg-1, res.device_params.jbeg-1}, {res.device_params.iend+1, res.device_params.jend+1});
  
  checkValidityIni(res);

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

  const real_t epsilon = full_params.epsilon_reset_negative;

  Kokkos::parallel_reduce(
    "Check negative density/pressure", 
    full_params.range_dom,
    KOKKOS_LAMBDA(const int i, const int j, uint64_t& lnegative_density, uint64_t& lnegative_pressure, uint64_t& lnan_count) {
      if (Q(j, i, IR) < 0) {
        Q(j, i, IR) = epsilon;
        lnegative_density++;
      }
      if (Q(j, i, IP) < 0) {
        Q(j, i, IP) = epsilon;
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

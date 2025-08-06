#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>
#include <filesystem>

#include "SimInfo.h"

using namespace H5Easy;

namespace fv2d {

constexpr int ite_nzeros = 4;
constexpr std::string_view ite_prefix = "ite_";

  // xdmf strings
namespace {
  char str_xdmf_header[] = R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
<!ENTITY file "%s:">
<!ENTITY fdim "%d %d">
<!ENTITY gdim "%d %d">
<!ENTITY GridEntity '
<Topology TopologyType="Quadrilateral" NodesPerElement="4" NumberOfElements="&fdim;">
  <DataItem Dimensions="&fdim; 4" NumberType="UInt" Precision="4" Format="HDF">&file;/connectivity</DataItem>
</Topology>
<Geometry GeometryType="XY">
  <DataItem Dimensions="&gdim; 2" NumberType="Float" Precision="8" Format="HDF">&file;/coordinates</DataItem>
</Geometry>'>
]>
<Xdmf Version="3.0">
<Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    )xml";
  #define format_xdmf_header(params, filename) \
          (filename).c_str(),                  \
          params.Ny,     params.Nx,            \
          params.Ny + 1, params.Nx + 1
  char str_xdmf_footer[] =
  R"xml(
  </Grid>
</Domain>
</Xdmf>)xml";

  char str_xdmf_ite_header[] =
  R"xml(
    <Grid Name="%s" GridType="Uniform">
      <Time Value="%.12lg" />
      &GridEntity;)xml";
  #define format_xdmf_ite_header(name, time) \
          (name).c_str(), time
  char str_xdmf_scalar_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
      </Attribute>)xml";
  #define format_xdmf_scalar_field(group, field) \
          field, (group).c_str(), field
  char str_xdmf_vector_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="&fdim; 2" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
      </Attribute>)xml";
  #define format_xdmf_vector_field(group, field) \
          field, (group).c_str(), field
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
    )xml";
  } // anonymous namespace

class IOManager {
  using Table = std::vector<real_t>;
  using Table2 = std::vector<std::array<real_t, 2>>;

public:
  Params params;
  DeviceParams &device_params;
  bool force_file_truncation = false;

  IOManager(Params &params)
    : params(params), device_params(params.device_params) 
    {
      if (!std::filesystem::exists(params.output_path)) {
        std::cout << "Output path does not exist, creating directory `" << params.output_path << "`." << std::endl;
        std::filesystem::create_directory(params.output_path);
      }
      
      std::ofstream out_ini_local("last.ini");
      params.reader.outputValues(out_ini_local);

      std::ofstream out_ini(params.output_path + "/" + params.filename_out + ".ini");
      params.reader.outputValues(out_ini);
    };

  ~IOManager() = default;

  enum SaveMethod { IO_UNIQUE, IO_MULTIPLE };
  
  void saveSolution(const Array &Q, int iteration, real_t t) {
    if (params.multiple_outputs)
      saveSolution_aux<IO_MULTIPLE>(Q, iteration, t);
    else
      saveSolution_aux<IO_UNIQUE>(Q, iteration, t);
  }

  template<enum SaveMethod save_method>
  void saveSolution_aux(const Array &Q, int iteration, real_t t)
  {
    constexpr bool is_multiple = (save_method == IO_MULTIPLE);
    std::ostringstream oss;
    
    std::string iteration_str, basename;
    if constexpr (is_multiple) {
      oss << params.filename_out << "_" << std::setw(ite_nzeros) << std::setfill('0') << iteration;
      iteration_str = oss.str();
      basename = oss.str();
    }
    else {
      oss << ite_prefix << std::setw(ite_nzeros) << std::setfill('0') << iteration;
      iteration_str = oss.str();
      basename = params.filename_out;
    }
    
    const std::string h5_filename  = basename + ".h5";
    const std::string xmf_filename = basename + ".xmf";
    const std::string output_path = params.output_path + "/";

    const bool write_attrs = (is_multiple || force_file_truncation || iteration == 0);
      
    auto flag_h5 = (write_attrs ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (write_attrs ? "w+" : "r+");
    File file(output_path + h5_filename, flag_h5);
    FILE* xdmf_fd = fopen((output_path + xmf_filename).c_str(), flag_xdmf);

    if (write_attrs) {
      force_file_truncation = false;
      file.createAttribute("Ntx", device_params.Ntx);
      file.createAttribute("Nty", device_params.Nty);
      file.createAttribute("Nx", device_params.Nx);
      file.createAttribute("Ny", device_params.Ny);
      file.createAttribute("ibeg", device_params.ibeg);
      file.createAttribute("iend", device_params.iend);
      file.createAttribute("jbeg", device_params.jbeg);
      file.createAttribute("jend", device_params.jend);
      file.createAttribute("problem", params.problem);
      
      Table2 coordinates;
      // -- vertex pos
      for (int j=device_params.jbeg; j <= device_params.jend; ++j) {
        for (int i=device_params.ibeg; i <= device_params.iend; ++i) {
          coordinates.push_back({
            (i-device_params.ibeg) * device_params.dx + device_params.xmin, 
            (j-device_params.jbeg) * device_params.dy + device_params.ymin 
          });
        }
      }
      file.createDataSet("coordinates", coordinates);

      std::vector<std::array<uint32_t, 4>> connectivity;
      // -- connectivity
      for (int j=device_params.jbeg; j < device_params.jend; ++j) {
        for (int i=device_params.ibeg; i < device_params.iend; ++i) {
          auto vertex_id = [&](int i, int j) -> uint32_t { return (i-device_params.ibeg) + (j-device_params.jbeg) * (device_params.Nx + 1); };
          connectivity.push_back({vertex_id(i, j), vertex_id(i+1, j), vertex_id(i+1, j+1), vertex_id(i, j+1)});
        }
      }
      file.createDataSet("connectivity", connectivity);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, h5_filename));
      if constexpr (!is_multiple) 
        fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tprs;
    Table2 tvel;

    for (int j=device_params.jbeg; j<device_params.jend; ++j) {
      for (int i=device_params.ibeg; i<device_params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tvel.push_back({u, v});
        tprs.push_back(p);
      }
    }

    auto save_groups = [&](auto &g) {
      g.createDataSet("rho", trho);
      g.createDataSet("velocity", tvel);
      g.createDataSet("prs", tprs);
      g.createAttribute("time", t);
      g.createAttribute("iteration", iteration);
    };

    if constexpr (is_multiple) {
      save_groups(file);
    }
    else {
      auto h5_group = file.createGroup(iteration_str);
      save_groups(h5_group);
    }

    std::string group;
    if constexpr (is_multiple) {
      group = "";
    }
    else {
      group = iteration_str + "/";
      fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    }

    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  RestartInfo loadSnapshot(Array &Q) {
    // example of unique_output restart_file: 'run.h5:/ite_0005'
    // or just 'run.h5' for the last iteration

    std::string restart_file = params.restart_file;
    std::string group = "";

    const auto delim_multi = restart_file.find(".h5:/");
    if (delim_multi != std::string::npos) {
      group = restart_file.substr(delim_multi + 5);
      restart_file = restart_file.substr(0, delim_multi + 3);
    }

    if ( !params.multiple_outputs && std::filesystem::equivalent(restart_file, params.output_path + "/" + params.filename_out + ".h5") ) {
      if (delim_multi != std::string::npos) {
        std::cerr << "Invalid restart file : if your restart file and output file are "
                     "the same, you can only start from the last iteration." << std::endl << std::endl;
        throw std::runtime_error("ERROR : Invalid restart_file.");
      }
    }
    else {
      this->force_file_truncation = true;
    }
    
    File file(restart_file, File::ReadOnly);
    real_t time;
    int iteration;

    if (file.hasAttribute("time")) {
      HighFive::Attribute attr_time = file.getAttribute("time");
      attr_time.read(time);
      HighFive::Attribute attr_ite = file.getAttribute("iteration");
      attr_ite.read(iteration);
    }
    else {
      if (group == "") {
        const size_t last_ite_index = file.getNumberObjects() - 1; // first groups are : coordinates and connectivity, then all the ite_xxxx. 
        group = file.getObjectName(last_ite_index);
      }
      HighFive::Group h5_group = file.getGroup(group);
      HighFive::Attribute attr_time = h5_group.getAttribute("time");
      attr_time.read(time);
      HighFive::Attribute attr_ite = h5_group.getAttribute("iteration");
      attr_ite.read(iteration);
      group = group + "/";
    }

    auto Nt = getShape(file, group + "rho")[0];

    if (Nt != device_params.Nx*device_params.Ny) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << device_params.Nx << "x" << device_params.Ny << "=" << device_params.Nx*device_params.Ny << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);

    std::cout << "Loading restart data from hdf5" << std::endl;

    auto load_and_copy = [&]<std::size_t N>(const std::string var_name, const std::array<IVar, N>& var_id) {
      using Elem  = std::conditional_t<N == 1, real_t, std::array<real_t, N>>;
      using Table = std::vector<Elem>;
      auto table = load<Table>(file, group + var_name);
      
      // Parallel for here ?
      int lid = 0;
      for (int y=0; y < device_params.Ny; ++y) {
        for (int x=0; x < device_params.Nx; ++x) {
          Elem elem = table[lid++];
          if constexpr (N == 1)
            Qhost(y+device_params.jbeg, x+device_params.ibeg, var_id[0]) = elem;
          else {
            for (int i=0; i<N; i++) {
              Qhost(y+device_params.jbeg, x+device_params.ibeg, var_id[i]) = elem[i];
            }
          }
        }
      }
    };
    load_and_copy("rho",      std::array{IR});
    load_and_copy("velocity", std::array{IU, IV});
    load_and_copy("prs",      std::array{IP});

    Kokkos::deep_copy(Q, Qhost);

    BoundaryManager bc(params);
    bc.fillBoundaries(Q);

    if (time + params.device_params.epsilon > params.tend) {
      std::cerr << "Restart time is greater than end time : " << std::endl
                << "  time: " << time << "\ttend: " << params.tend << std::endl << std::endl; 
      throw std::runtime_error("ERROR : restart time is greater than the end time.");
    }

    std::cout << "Restart finished !" << std::endl;

    if (force_file_truncation) {
      file.~File(); // free the h5 before saving
      saveSolution(Q, iteration, time);
    }

    return {time, iteration};
  }
};
}
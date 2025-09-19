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
<Topology TopologyType="2DSMesh" Dimensions="&gdim;"/>
<Geometry GeometryType="X_Y">
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/x</DataItem>
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/y</DataItem>
</Geometry>'>
]>
<Xdmf Version="3.0">
<Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    )xml";
  #define format_xdmf_header(params, filename) \
          (filename).c_str(),                  \
          ((params).write_ghost_cells ? (params).Nty : (params).Ny), \
          ((params).write_ghost_cells ? (params).Ntx : (params).Nx), \
          (((params).write_ghost_cells ? (params).Nty : (params).Ny) + 1), \
          (((params).write_ghost_cells ? (params).Ntx : (params).Nx) + 1)
          
  char str_xdmf_footer[] =
  R"xml(
  </Grid>
</Domain>
</Xdmf>)xml";

  char str_xdmf_ite_header[] =
  R"xml(
    <Grid Name="%s" GridType="Uniform">
      <Time Value="%lf" />
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
        <DataItem Dimensions="&fdim; 2" ItemType="Function" Function="JOIN($0, $1)">
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
        </DataItem>
      </Attribute>)xml";
  #ifndef MHD
    #define format_xdmf_vector_field(group, name, field_x, field_y) \
            name, (group).c_str(), field_x, (group).c_str(), field_y
    char str_xdmf_ite_footer[] =
    R"xml(
      </Grid>
      )xml";
  #else
        #define format_xdmf_vector_field(group, name, field_x, field_y, field_z) \
            name, (group).c_str(), field_x, (group).c_str(), field_y, (group).c_str(), field_z
    char str_xdmf_ite_footer[] =
    R"xml(
      </Grid>
      )xml";
  #endif //MHD
  } // anonymous namespace

class IOManager {
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

  void saveSolution(const Array &Q, int iteration, real_t t) {
    if (params.multiple_outputs)
      saveSolutionMultiple(Q, iteration, t);
    else
      saveSolutionUnique(Q, iteration, t);
  }

  void saveSolutionMultiple(const Array &Q, int iteration, real_t t)
  {
    std::ostringstream oss;
    
    oss << params.filename_out << "_" << std::setw(ite_nzeros) << std::setfill('0') << iteration;
    std::string iteration_str = oss.str();
    std::string h5_filename  = oss.str() + ".h5";
    std::string xmf_filename = oss.str() + ".xmf";
    std::string output_path = params.output_path + "/";

    File file(output_path + h5_filename, File::Truncate);
    FILE* xdmf_fd = fopen((output_path + xmf_filename).c_str(), "w+");

    int j0 = device_params.jbeg;
    int jN = device_params.jend;
    int i0 = device_params.ibeg;
    int iN = device_params.iend;
    if (device_params.write_ghost_cells){
      j0 = 0;
      jN += device_params.Ng; 
      i0 = 0;
      iN += device_params.Ng;
    }

    // number of cells (fdim) and number of vertices (gdim) used by XDMF
    int fNy = jN - j0; // cells in Y
    int fNx = iN - i0; // cells in X
    int gNy = fNy + 1; // vertices in Y
    int gNx = fNx + 1; // vertices in X

    file.createAttribute("Ntx",  device_params.Ntx);
    file.createAttribute("Nty",  device_params.Nty);
    file.createAttribute("Nx",   device_params.Nx);
    file.createAttribute("Ny",   device_params.Ny);
    file.createAttribute("ibeg", device_params.ibeg);
    file.createAttribute("iend", device_params.iend);
    file.createAttribute("jbeg", device_params.jbeg);
    file.createAttribute("jend", device_params.jend);
    file.createAttribute("problem", params.problem);

    std::vector<real_t> x, y;
    // -- vertex pos
    for (int j=j0; j <= jN; ++j) {
      for (int i=i0; i <= iN; ++i) {
        x.push_back((i-device_params.ibeg) * device_params.dx + device_params.xmin);
        y.push_back((j-device_params.jbeg) * device_params.dy + device_params.ymin);
      }
    }

    file.createDataSet("x", x);
    file.createDataSet("y", y);

    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    #ifdef MHD
    Table tw, tbx, tby, tbz, tdivB, tpsi;
    #endif

    for (int j=j0; j<jN; ++j) {
      for (int i=i0; i<iN; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tu.push_back(u);
        tv.push_back(v);
        tprs.push_back(p);

        #ifdef MHD
          real_t w  = Qhost(j, i, IW);
          real_t bx = Qhost(j, i, IBX);
          real_t by = Qhost(j, i, IBY);
          real_t bz = Qhost(j, i, IBZ);
          real_t psi = Qhost(j, i, IPSI);

          tw.push_back(w);
          tbx.push_back(bx);
          tby.push_back(by);
          tbz.push_back(bz);
          tpsi.push_back(psi);
          if (!device_params.write_ghost_cells){ // Special care to compute divB in this case, TODO later
            real_t dBx_dx = (Qhost(j, i+1, IBX) - Qhost(j, i-1, IBX)) / (2. * device_params.dx);
            real_t dBy_dy = (Qhost(j+1, i, IBY) - Qhost(j-1, i, IBY)) / (2. * device_params.dy);
            tdivB.push_back(dBx_dx + dBy_dy);
          }
          else
            tdivB.push_back(0.0);
        #endif //MHD
        }
      }

    file.createDataSet("rho", trho);
    file.createDataSet("u", tu);
    file.createDataSet("v", tv);
    file.createDataSet("prs", tprs);
    #ifdef MHD
      file.createDataSet("w", tw);
      file.createDataSet("bx", tbx);
      file.createDataSet("by", tby);
      file.createDataSet("bz", tbz);
      file.createDataSet("divB", tdivB);
      file.createDataSet("psi", tpsi);
    #endif //MHDÒ
    file.createAttribute("time", t);
    file.createAttribute("iteration", iteration);

    std::string group = "";

  fprintf(xdmf_fd, str_xdmf_header, h5_filename.c_str(), fNy, fNx, gNy, gNx);
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    #ifdef MHD
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v", "w"));
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "magnetic", "bx", "by", "bz"));    
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "divB"));
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "psi"));
    #else
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v"));
    #endif
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t) {
    std::ostringstream oss;
    
    oss << ite_prefix << std::setw(ite_nzeros) << std::setfill('0') << iteration;
    std::string iteration_str = oss.str();
    std::string h5_filename  = params.filename_out + ".h5";
    std::string xmf_filename = params.filename_out + ".xmf";
    std::string output_path = params.output_path + "/";

    force_file_truncation = (force_file_truncation || iteration == 0);
      
    auto flag_h5 = (force_file_truncation ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (force_file_truncation ? "w+" : "r+");
    File file(output_path + h5_filename, flag_h5);
    FILE* xdmf_fd = fopen((output_path + xmf_filename).c_str(), flag_xdmf);

    int j0 = device_params.jbeg;
    int jN = device_params.jend;
    int i0 = device_params.ibeg;
    int iN = device_params.iend;
    
    if (device_params.write_ghost_cells){
      j0 = 0;
      jN += device_params.Ng; 
      i0 = 0;
      iN += device_params.Ng;
    }
    if (force_file_truncation) {
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

      std::vector<real_t> x, y;
      // -- vertex pos
      for (int j=j0; j <= jN; ++j)
        for (int i=i0; i <= iN; ++i)
        {
          x.push_back((i-device_params.ibeg) * device_params.dx + device_params.xmin);
          y.push_back((j-device_params.jbeg) * device_params.dy + device_params.ymin);
        }
      file.createDataSet("x", x);
      file.createDataSet("y", y);

      // compute fdim/gdim for XDMF header (accounting for ghost cells if enabled)
      int fNy = jN - j0; // cells in Y
      int fNx = iN - i0; // cells in X
      int gNy = fNy + 1; // vertices in Y
      int gNx = fNx + 1; // vertices in X

      fprintf(xdmf_fd, str_xdmf_header, (params.filename_out + ".h5").c_str(), fNy, fNx, gNy, gNx);
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    #ifdef MHD
    Table tw, tbx, tby, tbz, tdivB, tpsi;
    #endif

    for (int j=j0; j<jN; ++j) {
      for (int i=i0; i<iN; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tu.push_back(u);
        tv.push_back(v);
        tprs.push_back(p);

        #ifdef MHD
        real_t w  = Qhost(j, i, IW);
        real_t bx = Qhost(j, i, IBX);
        real_t by = Qhost(j, i, IBY);
        real_t bz = Qhost(j, i, IBZ);
        real_t psi = Qhost(j, i, IPSI);

        tw.push_back(w);
        tbx.push_back(bx);
        tby.push_back(by);
        tbz.push_back(bz);
        tpsi.push_back(psi);
        if (!device_params.write_ghost_cells){
          real_t dBx_dx = (Qhost(j, i+1, IBX) - Qhost(j, i-1, IBX)) / (2. * device_params.dx);
          real_t dBy_dy = (Qhost(j+1, i, IBY) - Qhost(j-1, i, IBY)) / (2. * device_params.dy);
          tdivB.push_back(dBx_dx + dBy_dy);
        }
        else
          tdivB.push_back(0.0);
        #endif //MHD

      }
    }

    auto ite_group = file.createGroup(iteration_str);
    ite_group.createDataSet("rho", trho);
    ite_group.createDataSet("u", tu);
    ite_group.createDataSet("v", tv);
    ite_group.createDataSet("prs", tprs);
    #ifdef MHD
    ite_group.createDataSet("w", tw);
    ite_group.createDataSet("bx", tbx);
    ite_group.createDataSet("by", tby);
    ite_group.createDataSet("bz", tbz);
    ite_group.createDataSet("psi", tpsi);
    ite_group.createDataSet("divB", tdivB);
    #endif //MHD
    ite_group.createAttribute("time", t);
    ite_group.createAttribute("iteration", iteration);

    const std::string group = iteration_str + "/";

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    #ifndef MHD
        fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v"));
    #else
        fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v", "w"));
        fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "magnetic", "bx", "by", "bz"));
        fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "psi"));
        fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "divB"));
    #endif //MHD
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
        const size_t last_ite_index = file.getNumberObjects() - 3;
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
    using Table = std::vector<real_t>;

    std::cout << "Loading restart data from hdf5" << std::endl;

    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, group + var_name);
      // Parallel for here ?
      int lid = 0;
      for (int y=0; y < device_params.Ny; ++y) {
        for (int x=0; x < device_params.Nx; ++x) {
          Qhost(y+device_params.jbeg, x+device_params.ibeg, var_id) = table[lid++];
        }
      }
    };
    load_and_copy("rho", IR);
    load_and_copy("u",   IU);
    load_and_copy("v",   IV);
    load_and_copy("prs", IP);
    #ifdef MHD
      load_and_copy("w", IW);
      load_and_copy("bx", IBX);
      load_and_copy("by", IBY);
      load_and_copy("bz", IBZ);
      load_and_copy("psi", IPSI);
    #endif //MHD
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
#pragma once

#include "Geometry.h"

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>
#include <filesystem>

#include "SimInfo.h"
#include "Gradient.h"

#define SAVE_PRESSURE_GRADIENT

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
  #define format_xdmf_vector_field(group, name, field_x, field_y) \
          name, (group).c_str(), field_x, (group).c_str(), field_y
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
    )xml";
  } // anonymous namespace

class IOManager {
public:
  Params params;
  DeviceParams &device_params;
  Geometry geo;
  bool force_file_truncation = false;

  IOManager(Params &params)
    : params(params), device_params(params.device_params), geo(params.device_params) 
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
    x.reserve((device_params.Nx + 1) * (device_params.Ny + 1));
    y.reserve((device_params.Nx + 1) * (device_params.Ny + 1));
    // vertex position
    for (int j=device_params.jbeg; j <= device_params.jend; ++j)
      for (int i=device_params.ibeg; i <= device_params.iend; ++i)
      {
        Pos pos = geo.mapc2p_vertex(i, j); 
        x.push_back(pos[IX]);
        y.push_back(pos[IY]);
      }
    file.createDataSet("x", x);
    file.createDataSet("y", y);

    // center position
    x.clear(); y.clear();
    for (int j=device_params.jbeg; j < device_params.jend; ++j)
      for (int i=device_params.ibeg; i < device_params.iend; ++i)
      {
        Pos pos = geo.mapc2p_center(i, j); 
        x.push_back(pos[IX]);
        y.push_back(pos[IY]);
      }
    file.createDataSet("center_x", x);
    file.createDataSet("center_y", y);

    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    #ifdef SAVE_PRESSURE_GRADIENT
    // Pressure gradient
    Array gradP_dev = Array("gradP", device_params.Ny, device_params.Nx, 2);
    {
      auto geometry = this->geo; 
      auto gradient_type = device_params.gradient_type;
      Kokkos::parallel_for(
        "PressureGradient", 
        params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          auto getState = [](const Array& Q, int i, int j) -> real_t {return Q(j,i,IP);};
          Kokkos::Array<real_t, 2> grad = computeGradient(Q, getState, i, j, geometry, gradient_type);
          gradP_dev(j,i,IX) = grad[IX];
          gradP_dev(j,i,IY) = grad[IY];
        });
    }

    auto gradP = Kokkos::create_mirror(gradP_dev);
    Kokkos::deep_copy(gradP, gradP_dev);
    Table tgradP[2];
    #endif

    Table trho, tu, tv, tprs;
    for (int j=device_params.jbeg; j<device_params.jend; ++j) {
      for (int i=device_params.ibeg; i<device_params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tu.push_back(u);
        tv.push_back(v);
        tprs.push_back(p);
        
        #ifdef SAVE_PRESSURE_GRADIENT
        tgradP[IX].push_back(gradP(j, i, IX));
        tgradP[IY].push_back(gradP(j, i, IY));
        #endif
      }
    }

    file.createDataSet("rho", trho);
    file.createDataSet("u", tu);
    file.createDataSet("v", tv);
    file.createDataSet("prs", tprs);
    file.createAttribute("time", t);
    file.createAttribute("iteration", iteration);

    #ifdef SAVE_PRESSURE_GRADIENT
    file.createDataSet("dp_x", tgradP[IX]);
    file.createDataSet("dp_y", tgradP[IY]);
    #endif

    std::string group = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, h5_filename));
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    #ifdef SAVE_PRESSURE_GRADIENT
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "dp", "dp_x", "dp_y"));
    #endif
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
      x.reserve((device_params.Nx + 1) * (device_params.Ny + 1));
      y.reserve((device_params.Nx + 1) * (device_params.Ny + 1));

      // -- vertex pos
      for (int j=device_params.jbeg; j <= device_params.jend; ++j)
        for (int i=device_params.ibeg; i <= device_params.iend; ++i)
        {
          Pos pos = geo.mapc2p_vertex(i, j); // curvilinear
          x.push_back(pos[IX]);
          y.push_back(pos[IY]);
        }
      file.createDataSet("x", x);
      file.createDataSet("y", y);

      // -- center pos
      x.clear(); y.clear();
      for (int j=device_params.jbeg; j < device_params.jend; ++j)
        for (int i=device_params.ibeg; i < device_params.iend; ++i)
        {
          Pos pos = geo.mapc2p_center(i, j); // curvilinear
          x.push_back(pos[IX]);
          y.push_back(pos[IY]);
        }
      file.createDataSet("center_x", x);
      file.createDataSet("center_y", y);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, h5_filename));
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }

    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    #ifdef SAVE_PRESSURE_GRADIENT
    // Pressure gradient
    Array gradP_dev = Array("gradP", device_params.Nty, device_params.Ntx, 2);
    {
      auto geometry = this->geo; 
      auto &device_params = this->device_params;
      auto gradient_type = device_params.gradient_type;
      Kokkos::parallel_for(
        "PressureGradient", 
        params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {
          
          
          // auto getState = [](const Array& Q, int i, int j) -> real_t {return Q(j,i,IP);};
          // Kokkos::Array<real_t, 2> grad = computeGradient(Q, getState, i, j, geometry, gradient_type);
          // gradP_dev(j,i,IX) = grad[IX];
          // gradP_dev(j,i,IY) = grad[IY];


          const real_t cellArea = geometry.cellArea(i,j);
          real_t len;
          Pos lenL = geometry.getRotationMatrix(i, j, IX, ILEFT,  len); lenL = lenL * len;
          Pos lenR = geometry.getRotationMatrix(i, j, IX, IRIGHT, len); lenR = lenR * len;
          Pos lenD = geometry.getRotationMatrix(i, j, IY, ILEFT,  len); lenD = lenD * len;
          Pos lenU = geometry.getRotationMatrix(i, j, IY, IRIGHT, len); lenU = lenU * len;
          real_t r  = norm(geometry.mapc2p_center(i,j));
          real_t rl = norm(geometry.faceCenter(i, j, IX, ILEFT));
          real_t rr = norm(geometry.faceCenter(i, j, IX, IRIGHT));
          real_t rd = norm(geometry.faceCenter(i, j, IY, ILEFT));
          real_t ru = norm(geometry.faceCenter(i, j, IY, IRIGHT));
          // const real_t prs0   = Q(j, i, IP);
          real_t factor = 1;
          if      (device_params.wb_grav_factor == WBGF_PRS) factor = Q(j, i, IP) / device_params.spl_prs(r);
          else if (device_params.wb_grav_factor == WBGF_RHO) factor = Q(j, i, IR) / device_params.spl_rho(r);
          else if (device_params.wb_grav_factor == WBGF_PRS_RHO) factor = Q(j, i, IP) / Q(j, i, IR) * device_params.spl_rho(r) / device_params.spl_prs(r);
          
          const real_t prs0l = device_params.spl_prs(rl);
          const real_t prs0r = device_params.spl_prs(rr);
          const real_t prs0d = device_params.spl_prs(rd);
          const real_t prs0u = device_params.spl_prs(ru);

          real_t sx = factor * ( lenR[IX] * prs0r - lenL[IX] * prs0l + lenU[IX] * prs0u - lenD[IX] * prs0d) / cellArea;
          real_t sy = factor * ( lenR[IY] * prs0r - lenL[IY] * prs0l + lenU[IY] * prs0u - lenD[IY] * prs0d) / cellArea; 

          if (device_params.wb_grav_grad_correction) {
            auto getState = [&](const Array& Q, int i, int j) -> real_t {
              real_t r0  = norm(geometry.mapc2p_center(i,j));
              return Q(j,i,IP) / device_params.spl_prs(r0);
            };
            Kokkos::Array<real_t, 2> grad = computeGradient(Q, getState, i, j, geometry, gradient_type);
            const real_t beta0 = device_params.spl_prs(r);
            sx += beta0 * grad[IX];
            sy += beta0 * grad[IY];
          }

          gradP_dev(j,i,IX) = sx;
          gradP_dev(j,i,IY) = sy;
        });
    }

    auto gradP = Kokkos::create_mirror(gradP_dev);
    Kokkos::deep_copy(gradP, gradP_dev);
    Table tgradP[2];
    #endif

    Table t_rho_fluc, t_prs_fluc;
    Array bg_dev = Array("rho_fluc_dev", device_params.Nty, device_params.Ntx, 2);
    auto bg_host = Kokkos::create_mirror(bg_dev);
    {
      auto dparams = device_params;
      auto geometry = this->geo; 
      Kokkos::parallel_for(
        "PressureGradient", 
        params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j) {

          const real_t r = norm(geometry.mapc2p_center(i,j));

          bg_dev(j, i, 0) = Q(j, i, IR) / dparams.spl_rho(r);
          bg_dev(j, i, 1) = Q(j, i, IP) / dparams.spl_prs(r);
        });

    Kokkos::deep_copy(bg_host, bg_dev);
    }


    Table trho, tu, tv, tprs;
    for (int j=device_params.jbeg; j<device_params.jend; ++j) {
      for (int i=device_params.ibeg; i<device_params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tu.push_back(u);
        tv.push_back(v);
        tprs.push_back(p);

        #ifdef SAVE_PRESSURE_GRADIENT
        tgradP[IX].push_back(gradP(j, i, IX));
        tgradP[IY].push_back(gradP(j, i, IY));
        #endif

        t_rho_fluc.push_back(bg_host(j, i, 0));
        t_prs_fluc.push_back(bg_host(j, i, 1));
      }
    }

    auto ite_group = file.createGroup(iteration_str);
    ite_group.createDataSet("rho", trho);
    ite_group.createDataSet("u", tu);
    ite_group.createDataSet("v", tv);
    ite_group.createDataSet("prs", tprs);
    ite_group.createAttribute("time", t);
    ite_group.createAttribute("iteration", iteration);

    const std::string group = iteration_str + "/";

    #ifdef SAVE_PRESSURE_GRADIENT
    ite_group.createDataSet("dp_x", tgradP[IX]);
    ite_group.createDataSet("dp_y", tgradP[IY]);
    #endif

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    #ifdef SAVE_PRESSURE_GRADIENT
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "grad_prs", "dp_x", "dp_y"));
    #endif

    ite_group.createDataSet("rho_fluc", t_rho_fluc);
    ite_group.createDataSet("prs_fluc", t_prs_fluc);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho_fluc"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs_fluc"));

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

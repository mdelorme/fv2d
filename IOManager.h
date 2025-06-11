#pragma once

#include "Geometry.h"

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>

#include "SimInfo.h"
#include "Gradient.h"

#define SAVE_PRESSURE_GRADIENT

using namespace H5Easy;

namespace fv2d {

  // xdmf strings
  namespace {
    char str_xdmf_header[] = R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain CollectionType="Temporal">
    <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">

      <Topology Name="Main Topology" TopologyType="2DSMesh" NumberOfElements="%d %d"/>
      <Geometry Name="Main Geometry" GeometryType="X_Y">
        <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/x</DataItem>
        <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/y</DataItem>
      </Geometry>
      )xml";
    #define format_xdmf_header(params, path)                     \
           params.Ny + 1, params.Nx + 1,                         \
           params.Ny + 1, params.Nx + 1, (path + ".h5").c_str(), \
           params.Ny + 1, params.Nx + 1, (path + ".h5").c_str()

    char str_xdmf_footer[] =
    R"xml(
    </Grid>
  </Domain>
</Xdmf>)xml";

    char str_xdmf_ite_header[] =
    R"xml(
    <Grid Name="Cells" GridType="Uniform">
      <Time TimeType="Single" Value="%lf" />
      <Topology Reference="//Topology[@Name='Main Topology']" />
      <Geometry Reference="//Geometry[@Name='Main Geometry']" />)xml";

    char str_xdmf_scalar_field[] =
    R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
      </Attribute>)xml";
    #define format_xdmf_scalar_field(params, path, group, field) \
            field,                                               \
            params.Ny, params.Nx,                                \
            (path + ".h5").c_str(), group.c_str(), field

    char str_xdmf_vector_field[] =
    R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="%d %d 2" ItemType="Function" Function="JOIN($0, $1)">
          <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
        </DataItem>
      </Attribute>)xml";
    #define format_xdmf_vector_field(params, path, group, name, field_x, field_y) \
            name,                                                                 \
            params.Ny, params.Nx,                                                 \
            params.Ny, params.Nx,                                                 \
            (path + ".h5").c_str(), group.c_str(), field_x,                       \
            params.Ny, params.Nx,                                                 \
            (path + ".h5").c_str(), group.c_str(), field_y

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

  IOManager(Params &params)
    : params(params), device_params(params.device_params), geo(params.device_params) {};

  ~IOManager() = default;

  void saveSolution(const Array &Q, int iteration, real_t t, real_t dt) {
    if (params.multiple_outputs)
      saveSolutionMultiple(Q, iteration, t, dt);
    else
      saveSolutionUnique(Q, iteration, t, dt);
  }

  void saveSolutionMultiple(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << params.filename_out << "_" << std::setw(4) << std::setfill('0') << iteration;
    std::string path = oss.str();
    std::string h5_filename  = params.path_out + "/" + oss.str() + ".h5";
    std::string xmf_filename = params.path_out + "/" + oss.str() + ".xmf";

    File file(h5_filename, File::Truncate);
    FILE* xdmf_fd = fopen(xmf_filename.c_str(), "w+");

    file.createAttribute("Ntx",  device_params.Ntx);
    file.createAttribute("Nty",  device_params.Nty);
    file.createAttribute("Nx",   device_params.Nx);
    file.createAttribute("Ny",   device_params.Ny);
    file.createAttribute("ibeg", device_params.ibeg);
    file.createAttribute("iend", device_params.iend);
    file.createAttribute("jbeg", device_params.jbeg);
    file.createAttribute("jend", device_params.jend);
    file.createAttribute("problem", params.problem);
    file.createAttribute("iteration", iteration);
    
    {
      std::vector<real_t> x, y;
      x.reserve(device_params.Ntx * device_params.Nty);
      y.reserve(device_params.Ntx * device_params.Nty);

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
    }

    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    #ifdef SAVE_PRESSURE_GRADIENT
    // Pressure gradient
    Array gradP_dev = Array("gradP", device_params.Nty, device_params.Ntx, 2);
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

    #ifdef SAVE_PRESSURE_GRADIENT
    file.createDataSet("dp_x", tgradP[IX]);
    file.createDataSet("dp_y", tgradP[IY]);
    #endif

    std::string empty_string = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, path));
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, path, empty_string, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, path, empty_string, "velocity", "u", "v"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, path, empty_string, "prs"));
    #ifdef SAVE_PRESSURE_GRADIENT
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, path, empty_string, "dp", "dp_x", "dp_y"));
    #endif
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << "ite_" << std::setw(4) << std::setfill('0') << iteration;
    std::string path = oss.str();
    std::string filepath = params.path_out + "/" + params.filename_out;

    auto flag_h5 = (iteration == 0 ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (iteration == 0 ? "w+" : "r+");
    File file(filepath + ".h5", flag_h5);
    FILE* xdmf_fd = fopen((filepath + ".xdmf").c_str(), flag_xdmf);

    if (iteration == 0) {
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
      x.reserve(device_params.Ntx * device_params.Nty);
      y.reserve(device_params.Ntx * device_params.Nty);

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

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, params.filename_out));
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

    auto group = file.createGroup(path);
    group.createDataSet("rho", trho);
    group.createDataSet("u", tu);
    group.createDataSet("v", tv);
    group.createDataSet("prs", tprs);
    group.createAttribute("time", t);

    #ifdef SAVE_PRESSURE_GRADIENT
    group.createDataSet("dp_x", tgradP[IX]);
    group.createDataSet("dp_y", tgradP[IY]);
    #endif

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);

    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, params.filename_out, path, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, params.filename_out, path, "velocity", "u", "v"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, params.filename_out, path, "prs"));
    #ifdef SAVE_PRESSURE_GRADIENT
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, params.filename_out, path, "dp", "dp_x", "dp_y"));
    #endif
    
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  RestartInfo loadSnapshot(Array &Q) {
    File file(params.restart_file, File::ReadOnly);

    auto Nt = getShape(file, "rho")[0];

    if (Nt != device_params.Nx*device_params.Ny) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << device_params.Nx << "x" << device_params.Ny << "=" << device_params.Nx*device_params.Ny << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);
    using Table = std::vector<real_t>;

    std::cout << "Loading restart data from hdf5" << std::endl;

    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, var_name);
      // Parallel for here ?
      int lid = 0;
      for (int y=0; y < device_params.Ny; ++y) {
        for (int x=0; x < device_params.Nx; ++x) {
          Qhost(y+device_params.jbeg, x+device_params.ibeg, var_id) = table[lid++];
        }
      }
    };
    load_and_copy("rho", IR);
    load_and_copy("u", IU);
    load_and_copy("v", IV);
    load_and_copy("prs", IP);

    Kokkos::deep_copy(Q, Qhost);

    BoundaryManager bc(params);
    bc.fillBoundaries(Q);
    
    HighFive::Attribute attr_time = file.getAttribute("time");
    real_t time; attr_time.read(time);
    HighFive::Attribute attr_ite = file.getAttribute("iteration");
    int iteration; attr_ite.read(iteration);
    
    std::cout << "Restart finished !" << std::endl;

    return {time, iteration};
  }
};
}
#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>

#include "SimInfo.h"

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
    #define format_xdmf_header(params, path)                                          \
           params.Ny + 1, params.Nx + 1,                                        \
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
    #define format_xdmf_scalar_field(params, path, group, field)              \
            field,                                                      \
            params.Ny, params.Nx,                                       \
            (path + ".h5").c_str(), group.c_str(), field

    char str_xdmf_vector_field[] =
    R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="%d %d 2" ItemType="Function" Function="JOIN($0, $1)">
          <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
        </DataItem>
      </Attribute>)xml";
    #ifndef MHD
      #define format_xdmf_vector_field(params, path, group, name, field_x, field_y) \
              name,                                                           \
              params.Ny, params.Nx,                                           \
              params.Ny, params.Nx,                                           \
              (path + ".h5").c_str(), group.c_str(), field_x,  \
              params.Ny, params.Nx,                                           \
              (path + ".h5").c_str(), group.c_str(), field_y
    #else
      #define format_xdmf_vector_field(params, path, group, name, field_x, field_y, field_z) \
              name,                                                           \
              params.Ny, params.Nx,                                           \
              params.Ny, params.Nx,                                           \
              (path + ".h5").c_str(), group.c_str(), field_x,                 \
              params.Ny, params.Nx,                                           \
              (path + ".h5").c_str(), group.c_str(), field_y,                  \
              params.Ny, params.Nx,                                           \
              (path + ".h5").c_str(), group.c_str(), field_z
    #endif // MHD
    char str_xdmf_ite_footer[] =
    R"xml(
    </Grid>
    )xml";
  } // anonymous namespace

class IOManager {
public:
  Params params;
  BoundaryManager bc_manager;

  IOManager(Params &params)
    : params(params), bc_manager(params) {};

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
    std::string h5_filename  = oss.str() + ".h5";
    std::string xmf_filename = oss.str() + ".xmf";

    File file(h5_filename, File::Truncate);
    FILE* xdmf_fd = fopen(xmf_filename.c_str(), "w+");

    file.createAttribute("Ntx", params.Ntx);
    file.createAttribute("Nty", params.Nty);
    file.createAttribute("Nx", params.Nx);
    file.createAttribute("Ny", params.Ny);
    file.createAttribute("ibeg", params.ibeg);
    file.createAttribute("iend", params.iend);
    file.createAttribute("jbeg", params.jbeg);
    file.createAttribute("jend", params.jend);
    file.createAttribute("problem", params.problem);
    file.createAttribute("iteration", iteration);

    std::vector<real_t> x, y;
    // -- vertex pos
    for (int j=params.jbeg; j <= params.jend; ++j) {
      for (int i=params.ibeg; i <= params.iend; ++i) {
        x.push_back((i-params.ibeg) * params.dx);
        y.push_back((j-params.jbeg) * params.dy);
      }
    }

    file.createDataSet("x", x);
    file.createDataSet("y", y);

    using Table = std::vector<real_t>;
    bc_manager.fillBoundaries(Q);
    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    Table tw, tbx, tby, tbz; // TODO: Ajouter un condition sur régime MHD
    std::vector<Table> divB(params.Ntx, Table(params.Nty, 0.0));
    for (int j=params.jbeg; j<params.jend; ++j) {

      for (int i=params.ibeg; i<params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        trho.push_back(rho);
        tu.push_back(u);
        tv.push_back(v);
        tprs.push_back(p);

        if (mhd_run){
          real_t w  = Qhost(j, i, IW);
          real_t bx = Qhost(j, i, IBX);
          real_t by = Qhost(j, i, IBY);
          real_t bz = Qhost(j, i, IBZ);
          
          tw.push_back(w);
          tbx.push_back(bx);
          tby.push_back(by);
          tbz.push_back(bz);
          real_t dBx_dx = (Qhost(j+1, i, IBX) - Qhost(j-1, i, IBX)) / (2 * params.dx);
          real_t dBy_dy = (Qhost(j, i+1, IBY) - Qhost(j, i-1, IBY)) / (2 * params.dy);
          divB[i][j] = dBx_dx + dBy_dy;
        }
        }
      }

    file.createDataSet("rho", trho);
    file.createDataSet("u", tu);
    file.createDataSet("v", tv);
    file.createDataSet("prs", tprs);
    if (mhd_run) {
      file.createDataSet("w", tw);
      file.createDataSet("bx", tbx);
      file.createDataSet("by", tby);
      file.createDataSet("bz", tbz);
      file.createDataSet("divB", divB);
    }
    file.createAttribute("time", t);

    std::string empty_string = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(params, path));
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, empty_string, "rho"));
    #ifdef MHD
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, path, empty_string, "velocity", "u", "v", "w"));
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, path, empty_string, "magnetic_field", "bx", "by", "bz"));    
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, empty_string, "divB"));
    #else
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, path, empty_string, "velocity", "u", "v"));
    #endif
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, empty_string, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << "ite_" << std::setw(4) << std::setfill('0') << iteration;
    std::string path = oss.str();

    auto flag_h5 = (iteration == 0 ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (iteration == 0 ? "w+" : "r+");
    File file(params.filename_out + ".h5", flag_h5);
    FILE* xdmf_fd = fopen((params.filename_out + ".xdmf").c_str(), flag_xdmf);

    if (iteration == 0) {
      file.createAttribute("Ntx", params.Ntx);
      file.createAttribute("Nty", params.Nty);
      file.createAttribute("Nx", params.Nx);
      file.createAttribute("Ny", params.Ny);
      file.createAttribute("ibeg", params.ibeg);
      file.createAttribute("iend", params.iend);
      file.createAttribute("jbeg", params.jbeg);
      file.createAttribute("jend", params.jend);
      file.createAttribute("problem", params.problem);

      std::vector<real_t> x, y;
      // -- vertex pos
      for (int j=params.jbeg; j <= params.jend; ++j)
        for (int i=params.ibeg; i <= params.iend; ++i)
        {
          x.push_back((i-params.ibeg) * params.dx);
          y.push_back((j-params.jbeg) * params.dy);
        }
      file.createDataSet("x", x);
      file.createDataSet("y", y);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(params, params.filename_out));
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }

    using Table = std::vector<std::vector<real_t>>;
    bc_manager.fillBoundaries(Q);
    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    Table tw, tbx, tby, tbz; // TODO: Ajouter un condition sur régime MHD
    Table divB(params.Ntx, std::vector<real_t>(params.Nty, 0.0));
    for (int j=params.jbeg; j<params.jend; ++j) {
      std::vector<real_t> rrho, ru, rv, rprs;
      std::vector<real_t> rw, rbx, rby, rbz; // TODO: Ajouter un condition sur régime MHD
      for (int i=params.ibeg; i<params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        rrho.push_back(rho);
        ru.push_back(u);
        rv.push_back(v);
        rprs.push_back(p);

        if (mhd_run){
        real_t w  = Qhost(j, i, IW);
        real_t bx = Qhost(j, i, IBX);
        real_t by = Qhost(j, i, IBY);
        real_t bz = Qhost(j, i, IBZ);
        
        rw.push_back(w);
        rbx.push_back(bx);
        rby.push_back(by);
        rbz.push_back(bz);
        real_t dBx_dx = (Qhost(j+1, i, IBX) - Qhost(j-1, i, IBX)) / (2 * params.dx);
        real_t dBy_dy = (Qhost(j, i+1, IBY) - Qhost(j, i-1, IBY)) / (2 * params.dy);
        divB[i][j] = dBx_dx + dBy_dy;
        }
      }
      trho.push_back(rrho);
      tu.push_back(ru);
      tv.push_back(rv);
      tprs.push_back(rprs);

      if (mhd_run){
      tw.push_back(rw);
      tbx.push_back(rbx);
      tby.push_back(rby);
      tbz.push_back(rbz);
      }
  } // loop j

    auto group = file.createGroup(path);
    group.createDataSet("rho", trho);
    group.createDataSet("u", tu);
    group.createDataSet("v", tv);
    group.createDataSet("prs", tprs);
    if (mhd_run){
      group.createDataSet("w", tw); // TODO: Ajouter un condition sur régime MHD
      group.createDataSet("bx", tbx);
      group.createDataSet("by", tby);
      group.createDataSet("bz", tbz);
      group.createDataSet("divB", divB);
    }
    group.createAttribute("time", t);

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, params.filename_out, path, "rho"));
    #ifdef MHD
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, params.filename_out, path, "velocity", "u", "v", "w"));
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, params.filename_out, path, "magnetic_field", "bx", "by", "bz"));
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, params.filename_out, path, "divB"));
    #else
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, params.filename_out, path, "velocity", "u", "v"));
    #endif
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, params.filename_out, path, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  RestartInfo loadSnapshot(Array &Q) {
    File file(params.restart_file, File::ReadOnly);

    auto Nt = getShape(file, "rho")[0];

    if (Nt != params.Nx*params.Ny) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << params.Nx << "x" << params.Ny << "=" << params.Nx*params.Ny << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);
    using Table = std::vector<real_t>;

    std::cout << "Loading restart data from hdf5" << std::endl;

    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, var_name);
      // Parallel for here ?
      int lid = 0;
      for (int y=0; y < params.Ny; ++y) {
        for (int x=0; x < params.Nx; ++x) {
          Qhost(y+params.jbeg, x+params.ibeg, var_id) = table[lid++];
        }
      }
    };
    load_and_copy("rho", IR);
    load_and_copy("u", IU);
    load_and_copy("v", IV);
    load_and_copy("prs", IP);
    if (mhd_run){
      load_and_copy("w", IW);
      load_and_copy("bx", IBX);
      load_and_copy("by", IBY);
      load_and_copy("bz", IBZ);
    }
    Kokkos::deep_copy(Q, Qhost);

    std::cout << "Restart finished !" << std::endl;

    real_t time = loadAttribute<real_t>(file, "", "time");
    int iteration = loadAttribute<int>(file, "", "iteration");

    return {time, iteration};
  }
};
}
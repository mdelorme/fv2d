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
    #define format_xdmf_vector_field(params, path, group, name, field_x, field_y) \
            name,                                                           \
            params.Ny, params.Nx,                                           \
            params.Ny, params.Nx,                                           \
            (path + ".h5").c_str(), group.c_str(), field_x,  \
            params.Ny, params.Nx,                                           \
            (path + ".h5").c_str(), group.c_str(), field_y

    char str_xdmf_ite_footer[] =
    R"xml(
    </Grid>
    )xml";
  } // anonymous namespace

class IOManager {
public:
  Params params;

  IOManager(Params &params)
    : params(params) {};

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

    using Table = std::vector<std::vector<real_t>>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    for (int j=params.jbeg; j<params.jend; ++j) {
      std::vector<real_t> rrho, ru, rv, rprs;

      for (int i=params.ibeg; i<params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        rrho.push_back(rho);
        ru.push_back(u);
        rv.push_back(v);
        rprs.push_back(p);
      }

      trho.push_back(rrho);
      tu.push_back(ru);
      tv.push_back(rv);
      tprs.push_back(rprs);
    }

    file.createDataSet("rho", trho);
    file.createDataSet("u", tu);
    file.createDataSet("v", tv);
    file.createDataSet("prs", tprs);
    file.createAttribute("time", t);

    std::string empty_string = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(params, path));
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, empty_string, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, path, empty_string, "velocity", "u", "v"));
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

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tprs;
    for (int j=params.jbeg; j<params.jend; ++j) {
      std::vector<real_t> rrho, ru, rv, rprs;

      for (int i=params.ibeg; i<params.iend; ++i) {
        real_t rho = Qhost(j, i, IR);
        real_t u   = Qhost(j, i, IU);
        real_t v   = Qhost(j, i, IV);
        real_t p   = Qhost(j, i, IP);

        rrho.push_back(rho);
        ru.push_back(u);
        rv.push_back(v);
        rprs.push_back(p);
      }

      trho.push_back(rrho);
      tu.push_back(ru);
      tv.push_back(rv);
      tprs.push_back(rprs);
    }

    auto group = file.createGroup(path);
    group.createDataSet("rho", trho);
    group.createDataSet("u", tu);
    group.createDataSet("v", tv);
    group.createDataSet("prs", tprs);
    group.createAttribute("time", t);

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, params.filename_out, path, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, params.filename_out, path, "velocity", "u", "v"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, params.filename_out, path, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }
};

}
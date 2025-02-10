#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>

#include "SimInfo.h"

using namespace H5Easy;

namespace fv2d {

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
    std::string h5_filename = oss.str() + ".h5";

    File file(h5_filename, File::Truncate);

    file.createAttribute("Ntx", params.Ntx);
    file.createAttribute("Nty", params.Nty);
    file.createAttribute("Nx", params.Nx);
    file.createAttribute("Ny", params.Ny);
    file.createAttribute("ibeg", params.ibeg);
    file.createAttribute("iend", params.iend);
    file.createAttribute("jbeg", params.jbeg);
    file.createAttribute("jend", params.jend);
    file.createAttribute("problem", params.problem);

    std::vector<real_t> x;
    for (int i=params.ibeg; i < params.iend; ++i)
      x.push_back(((i-params.ibeg)+0.5) * params.dx);
    file.createDataSet("x", x);
    std::vector<real_t> y;
    for (int j=params.jbeg; j < params.jend; ++j)
      y.push_back(((j-params.jbeg)+0.5) * params.dy);
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
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << "ite_" << std::setw(4) << std::setfill('0') << iteration;
    std::string path = oss.str();
    auto flag = (iteration == 0 ? File::Truncate : File::ReadWrite);

    File file(params.filename_out + ".h5", flag);

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

      std::vector<real_t> x;
      for (int i=params.ibeg; i < params.iend; ++i)
        x.push_back(((i-params.ibeg)+0.5) * params.dx);
      file.createDataSet("x", x);
      std::vector<real_t> y;
      for (int j=params.jbeg; j < params.jend; ++j)
        y.push_back(((j-params.jbeg)+0.5) * params.dy);
      file.createDataSet("y", y);
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
  }
};

}
#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>

#include "SimInfo.h"

using namespace H5Easy;

namespace fv1d {

void save_solution(const Array &Q, int iteration, real_t t, real_t dt) {
  std::ostringstream oss;
  
  std::setw(4);
  std::setfill('0');
  oss << "ite_" << iteration;
  std::string path = oss.str();
  auto flag = (iteration == 0 ? File::Truncate : File::ReadWrite);

  File file(filename_out, flag);

  if (iteration == 0) {
    file.createAttribute("Ntx", Ntx);
    file.createAttribute("Nty", Nty);
    file.createAttribute("Nx", Nx);
    file.createAttribute("Ny", Ny);
    file.createAttribute("ibeg", ibeg);
    file.createAttribute("iend", iend);
    file.createAttribute("jbeg", jbeg);
    file.createAttribute("jend", jend);
    file.createAttribute("problem", problem);

    std::vector<real_t> x;
    for (int i=ibeg; i < iend; ++i)
      x.push_back((i+0.5) * dx);
    file.createDataSet("x", x);
    std::vector<real_t> y;
    for (int j=jbeg; j < jend; ++j)
      y.push_back((j+0.5) * dy);
    file.createDataSet("y", y);
  }

  

  using Table = std::vector<std::vector<real_t>>;

  Table trho, tu, tv, tprs;
  for (int j=jbeg; j<jend; ++j) {
    std::vector<real_t> rrho, ru, rv, rprs;

    for (int i=ibeg; i<iend; ++i) {
      real_t rho = Q[j][i][IR];
      real_t u   = Q[j][i][IU];
      real_t v   = Q[j][i][IV];
      real_t Ek = 0.5 * rho * (u*u + v*v);
      real_t p = Q[j][i][IP];

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

}
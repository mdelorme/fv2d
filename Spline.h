#pragma once

#include "SimInfo.h"
#include <fstream>

namespace fv2d {

class Spline {
// using real_t = double;

public:
  enum ReadfileOffset { OF_RHO, OF_PRS, OF_GRAVITY, OF_KAPPA, OF_HEATING, };
  Spline() = default;

  Spline(std::string filename, ReadfileOffset off)
  {
    std::ifstream data_file(filename, std::ios::in|std::ios::binary);
    if (!data_file) {
      std::cout << "cannot open file: " << filename << std::endl;
      Kokkos::abort("");
    }

    data_file.read(reinterpret_cast<char*>(&header), sizeof(header));
    data_file.seekg(header.offset_variable[off]);

    Kokkos::View<real_t*[4], Kokkos::DefaultHostExecutionSpace> 
      spline_host("spline_host", header.N * sizeof(real_t[4]));
    data_file.read(reinterpret_cast<char*>(spline_host.data()), header.N * sizeof(real_t[4]));

    // auto device_view_mirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), spline_host);
    
    // spline = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), spline_host);

    spline = Kokkos::View<real_t*[4], Kokkos::DefaultExecutionSpace>("spline", header.N * sizeof(real_t[4]));
    auto host_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), spline_host);

    Kokkos::deep_copy(spline, host_mirror);

    data_file.close();
  }

  KOKKOS_INLINE_FUNCTION
  real_t operator()(real_t r) const 
  {
    return this->GetValue(r);
  }

  KOKKOS_INLINE_FUNCTION
  real_t GetValue(real_t r) const
  {
    const int32_t N = static_cast<int32_t>(header.N);
    const real_t r0 = header.r0;
    const real_t dr = header.dr;


    int32_t id = (r-r0)/dr;
    id = (id <  0) ?   0 : id;
    id = (id >= N) ? N-1 : id;

    const real_t c[4] = {spline(id, 0), spline(id, 1), spline(id, 2), spline(id, 3)};
    const real_t rs = r0 + id * dr;
    const real_t ri = r-rs;

    return c[0] + ri * c[1] + ri*ri * c[2] + ri*ri*ri * c[3];
  }

private:
  struct {
    uint32_t N;
    uint32_t offset_variable[5];
    real_t r0, dr, rcut;
    real_t Cp, R, gamma;
  } header;
  Kokkos::View<real_t*[4]> spline;
};

}

#pragma once

#include <Kokkos_Core.hpp>

#include <fstream>

namespace fv2d {

class Profile {
public:
  enum ProfileVar {
    IU=0,
    IV=4,
    IRHO=8,
    IP=12,
    IKAPPA=16,
    IGRAVITY=20,
    IMU=24,
  };

private:
  Kokkos::View<real_t*> breakpoints;
  Kokkos::View<real_t**> values;
  size_t N;
  real_t ymin, ymax;
  const size_t ncol=IMU+4;
  const int ncoeff = 4;

  void readFromHDF5(std::string filename) {
    using namespace H5Easy;
    
    File file(filename, File::ReadOnly);

    using Table = std::vector<real_t>;
    using TwoDimTable = std::vector<Table>;
    const std::vector<std::string> field_names({"u_spline", 
                                                "v_spline", 
                                                "rho_spline", 
                                                "p_spline", 
                                                "kappa_rad_spline", 
                                                "gravity_spline",
                                                "mu_spline"});
    std::map<std::string, ProfileVar> field_map{
      {"u_spline",           IU},
      {"v_spline",           IV},
      {"rho_spline",         IRHO},
      {"p_spline",           IP},
      {"kappa_rad_spline",   IKAPPA},
      {"gravity_spline",     IGRAVITY},
      {"mu_spline",          IMU}
    };

    // Reading the breakpoints 
    if (!file.exist("y")) {
      std::ostringstream error_msg;
      error_msg << "ERROR ! You must provide a breakpoint field 'y' with the input profiles !";
      throw std::runtime_error(error_msg.str());
    }
    Table v = load<Table>(file, "y");
    N = v.size(); 

    // Allocating the views
    breakpoints = Kokkos::View<real_t*>("Profile", N);
    auto breakpoints_host = Kokkos::create_mirror_view(breakpoints);
    values = Kokkos::View<real_t**>("Profile", N-1, ncol);
    auto values_host = Kokkos::create_mirror_view(values);

    for (size_t i=0; i < N; ++i)
      breakpoints_host(i) = v[i]; 

    // Reading the hdf5 file
    for (auto &f: field_names) {
      if (file.exist(f)) {
        size_t ivar = field_map[f];
        TwoDimTable v = load<TwoDimTable>(file, f);
        if (v[0].size() != N-1) { 
          std::ostringstream error_msg;
          error_msg << "ERROR ! Loading profile " << filename << "; Fields are inconsistent with the breakpoint length !";
          throw std::runtime_error(error_msg.str());
        }
        for (size_t i=0; i < N; ++i) {
          for (int j=0; j < ncoeff; ++j) {
            values_host(i, ivar+j) = v[j][i];
          }
        }
      }
      else {
        std::cout << "Warning ! Loading profile " << filename << "; Field " << f << " is not stored in the file. Skipping." << std::endl;
      }
    }

    // Pushing to device
    Kokkos::deep_copy(values, values_host);
    Kokkos::deep_copy(breakpoints, breakpoints_host);

    ymin = breakpoints_host(0);
    ymax = breakpoints_host(N-1);

  }

public:
  Profile()  = default;
  ~Profile() = default;

  /**
   * @brief Reads the profile from a file
   * 
   * The file has to be hdf5.
   * It is necessary to provide a file with all variables given as a datasets in the root.
   */
  void readFromFile(std::string filename) {
    if (filename.ends_with(".h5"))
      readFromHDF5(filename);
    else
      throw std::runtime_error("Unsupported file format for profile " + filename);
    std::cout << "Read profile from \"" << filename << "\". Profile has " << N << " entries." << std::endl;
  }

  /**
   * @brief Returns the closes index to a given position
   */
  KOKKOS_INLINE_FUNCTION
  int getClosestLowerIndex(real_t yval) const {
    if (yval <= ymin)
      return 0;
    if (yval >= ymax)
      return N-1;

    // Dichotomic search
    int low = 0;
    int high = N-1;
    while (low < high-1) {
      int mid = (low + high) / 2;
      if (yval > breakpoints(mid))
        low = mid;
      else
        high = mid;
    }
    return low;
  }

  /**
   * @brief Returns a value at the given position computed from the
   * input spline.
   */
  KOKKOS_INLINE_FUNCTION
  real_t compute_from_spline(real_t yval, ProfileVar ivar) const {
    int i = getClosestLowerIndex(yval);
    const real_t ylow  = breakpoints(i);
    real_t val = 0;
    for (int k=0; k < ncoeff; ++k) {
      val = val + values(i, ivar+k) * pow(yval-ylow, 3-k);
    }
    return val;
  }

};
}
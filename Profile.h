#pragma once

#include <Kokkos_Core.hpp>

#include <fstream>

namespace fv2d {

class Profile {
public:
  enum ProfileVar {
    IY,
    IRHO,
    IU,
    IV,
    IP,
    IKAPPA,
    IGRAVITY
  };

private:
  Kokkos::View<real_t**> values;
  size_t N;
  real_t ymin, ymax;

  void readFromHDF5(std::string filename) {
    using namespace H5Easy;
    
    File file(filename, File::ReadOnly);

    // Storing all the data in a map
    using Table = std::vector<real_t>;
    std::map<std::string, Table> all_data;

    const std::vector<std::string> field_names({"y", "rho", "u", "v", "p", "kappa_rad", "gravity"});
    
    // Reading all the data and storing it in the map
    N = 0; 
    for (auto &f: field_names) {
      if (file.exist(f)) {
        Table v = load<Table>(file, f);
        all_data[f] = v;
        if (N == 0)
          N = v.size();
        else if (N != v.size()) {
          std::ostringstream error_msg;
          error_msg << "ERROR ! Loading profile " << filename << "; Fields are inconsistent and not having the same lengths !";
          throw std::runtime_error(error_msg.str());
        }
      }
      else {
        std::cout << "Warning ! Loading profile " << filename << "; Field " << f << " is not stored in the file. Skipping." << std::endl;
      }
    }

    // Allocating the view
    values = Kokkos::View<real_t**>("Profile", N, field_names.size());
    auto values_host = Kokkos::create_mirror_view(values);

    // Copying the data
    size_t ivar=0;
    for (auto &f: field_names) {
      // Copying only if the field is loaded
      if (all_data.count(f) != 0) {
        for (size_t i=0; i < N; ++i)
          values_host(i, ivar) = all_data[f][i];
      }
      ivar++;
    }

    // Pushing to device
    Kokkos::deep_copy(values, values_host);

    // Outputting fields if required
    auto f_out = std::ofstream("profile_table.txt");

    // Printing field names
    f_out << "#";
    for (auto &f: field_names)
      f_out << f << " ";
    f_out << std::endl;

    const size_t nv = field_names.size();
    for (size_t i=0; i < N; ++i) {
      for (size_t v=0; v < nv; ++v) 
        f_out << values_host(i, v) << " ";
      f_out << std::endl;
    }
    f_out.close();

  }

public:
  Profile()  = default;
  ~Profile() = default;

  /**
   * @brief Reads the profile from a file
   * 
   * The file can be either text or hdf5.
   * In the case of a text file, columns are expected to be separated by spaces or tabs
   * in the case of an hdf5 it is necessary to provide a file with all variables given asa datasets in the root.
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
    return N * (yval-ymin) / (ymax-ymin);
  }

  /**
   * @brief Returns the value of the given variable at the given position
   */
  KOKKOS_INLINE_FUNCTION
  real_t at(int j, ProfileVar ivar) const {
    return values(j, ivar);
  }

  /**
   * @brief Returns a linearly interpolated value at the given position
   */
  KOKKOS_INLINE_FUNCTION
  real_t interpolate_at(real_t yval, ProfileVar ivar) const {
    int i = getClosestLowerIndex(yval);
    if (i < 0 )
      return values(0, IY);
    if (i >= N)
      return values(N-1, IY);

    const real_t ylow  = values(i, IY);
    const real_t yhigh = values(i+1, IY);
    const real_t x = (yval-ylow) / (yhigh-ylow);
    const real_t vlow  = values(i, ivar);
    const real_t vhigh = values(i+1, ivar);
    return vlow * (1-x) + vhigh * x;
  }
};
}
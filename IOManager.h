#pragma once

#include <highfive/H5Easy.hpp>
#include <filesystem>

#include "SimInfo.h"

using namespace H5Easy;

namespace fv2d {

constexpr int ite_nzeros = 4;

consteval auto io_entry(std::string_view name, auto... ivars) { return std::make_pair(name, std::to_array({ ivars... })); }

// register output variables : name & indices
constexpr auto io_variables = std::make_tuple(
  io_entry("rho",      IR),
  io_entry("velocity", IU, IV),
  io_entry("prs",      IP)
);

  // xdmf strings
namespace {
constexpr std::string_view str_xdmf_footer_unique = R"xml(</Grid>
</Domain>
</Xdmf>)xml";
constexpr std::string_view str_xdmf_main_footer = R"xml(
  </Grid>
</Domain>
</Xdmf>)xml";
}

class IOManager {
  template<std::size_t N>
  using Table = std::vector<std::array<real_t, N>>;

  template<std::size_t I, typename F, typename... Tuples>
  constexpr void call_index(F&& func, Tuples&&... tuples) {
    std::invoke(std::forward<F>(func), std::get<I>(std::forward<Tuples>(tuples))...);
  }

  template<typename F, typename... Tuples, std::size_t... I>
  constexpr void for_each_tuples_aux(F&& func, std::index_sequence<I...>, Tuples&&... tuples) {
    ( (call_index<I>(std::forward<F>(func), std::forward<Tuples>(tuples)...)), ... );
  }

  template<typename F, typename... Tuples>
  constexpr void for_each_tuples(F&& func, Tuples&&... tuples) {
    using FirstTuple = std::remove_reference_t<std::tuple_element_t<0, std::tuple<Tuples...>>>;
    constexpr std::size_t N = std::tuple_size_v<FirstTuple>;
    for_each_tuples_aux(std::forward<F>(func), std::make_index_sequence<N>{}, std::forward<Tuples>(tuples)...);
  }

  template<std::size_t... I>
  constexpr auto make_tables_aux(std::index_sequence<I...>) {
    using Tabs = std::tuple<Table<std::tuple_size_v<decltype(std::get<I>(io_variables).second)>>...>;
    Tabs tabs;
    return tabs;
  } 

  auto make_tables() {
    const std::size_t capacity = device_params.Nx * device_params.Ny;
    constexpr std::size_t N = std::tuple_size_v<decltype(io_variables)>;
    auto tabs{make_tables_aux(std::make_index_sequence<N>{})};
    std::apply([capacity](auto&... vec){ (vec.reserve(capacity), ...) ;}, tabs);
    return tabs;
  }

  template<std::size_t... I>
  constexpr auto tuple_repeat(std::index_sequence<I...>, const auto&... t) {
    return std::tuple_cat((static_cast<void>(I), std::make_tuple(t...))...);
  } 

  auto get_xdmf_format_tuple(const char* grid_name, real_t time, const char* file, const char* group) {
    constexpr std::size_t N = std::tuple_size_v<decltype(io_variables)>;
    const auto head = std::make_tuple(xdmf_str.c_str(), grid_name, time, file, file);
    const auto tail = tuple_repeat(std::make_index_sequence<N>{}, file, group);
    return std::tuple_cat(head, tail);
  }

  std::string format_string(auto&&... args) {
    const int buf_size = std::snprintf(nullptr, 0, args...);
    std::string str(buf_size + 1, '\0');
    std::snprintf(str.data(), buf_size + 1, args...);
    str.resize(buf_size);
    return str;
  }
  
  template<typename... T> // overload for tuple
  std::string format_string(const std::tuple<T...>& tuple) {
    return std::apply([&](auto&&... args) -> std::string { return format_string(std::forward<decltype(args)>(args)...); }, tuple);
  }

  std::string init_xdmf_str() {
    std::ostringstream o;
    const std::string field_size_str = format_string("%d %d", device_params.Ny,   device_params.Nx);
    const std::string grid_size_str  = format_string("%d %d", device_params.Ny+1, device_params.Nx+1);

    if (params.multiple_outputs) {
      o << R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
<Domain>)xml";
    }

    constexpr std::string_view ite_begin = R"xml(
  <Grid Name="%%s" GridType="Uniform">
    <Time Value="%%.12lg" />
    <Topology TopologyType="Quadrilateral" NodesPerElement="4" NumberOfElements="%s">
      <DataItem Dimensions="%s 4" NumberType="UInt" Precision="4" Format="HDF">%%s:/connectivity</DataItem>
    </Topology>
    <Geometry GeometryType="XY">
      <DataItem Dimensions="%s 2" NumberType="Float" Precision="%lu" Format="HDF">%%s:/coordinates</DataItem>
    </Geometry>)xml";

    o << format_string(ite_begin.data(), field_size_str.c_str(), field_size_str.c_str(), grid_size_str.c_str(), sizeof(real_t));

    for_each_tuples([&](const auto &io_var){
      constexpr std::size_t dim = std::tuple_size_v<typename std::remove_reference_t<decltype(io_var)>::second_type>;
      constexpr std::string_view field_type = (dim > 3) ? "Matrix" : (dim > 1) ? "Vector" : "Scalar";
      const std::string_view name = io_var.first;
      
      constexpr std::string_view xdmf_field_str = R"xml(
    <Attribute Name="%s" AttributeType="%s" Center="Cell">
      <DataItem Dimensions="%s %lu" NumberType="Float" Precision="%lu" Format="HDF">%%s:/%%s%s</DataItem>
    </Attribute>)xml";
      
      o << format_string(xdmf_field_str.data(), name.data(), field_type.data(), field_size_str.c_str(), dim, sizeof(real_t), name.data());
    }, io_variables);

    o << R"xml(
  </Grid>
)xml";
    
    if (params.multiple_outputs) {
      o << R"xml(</Domain>
</Xdmf>)xml";
    }
    else {
      o << str_xdmf_footer_unique.data();
    }
    return o.str();
  }

public:
  Params params;
  DeviceParams &device_params;
  bool first_iteration = true;
  const std::string xdmf_str;

  IOManager(Params &params)
    : params(params), device_params(params.device_params), xdmf_str(init_xdmf_str())
    {
      if (!std::filesystem::exists(params.output_path)) {
        std::cout << "Output path does not exist, creating directory `" << params.output_path << "`." << std::endl;
        std::filesystem::create_directory(params.output_path);
      }
      
      std::ofstream out_ini_local("last.ini");
      params.reader.outputValues(out_ini_local);

      std::ofstream out_ini(params.output_path + params.filename_out + ".ini");
      params.reader.outputValues(out_ini);

      if (params.restart_file == "") 
      {
        // initialize xdmf for unique output
        if (!params.multiple_outputs) {
          std::ofstream xdmf_file_unique(params.output_path + params.filename_out + ".xmf", std::fstream::trunc);
          xdmf_file_unique << R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
<Domain>
<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
  )xml" << str_xdmf_footer_unique.data();
        }

        // initialize xdmf main for multiple output
        else {
          std::ofstream xdmf_main_file(params.output_path + params.filename_out + "_main.xmf", std::fstream::trunc);
          xdmf_main_file << R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
<Domain Name="MainTimeSeries">
  <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">)xml" << str_xdmf_main_footer.data();
        }
      }
    };

  ~IOManager() = default;

  enum SaveMethod { IO_UNIQUE, IO_MULTIPLE };
  
  void saveSolution(const Array &Q, int iteration, real_t t) {
    if (params.multiple_outputs)
      saveSolution_aux<IO_MULTIPLE>(Q, iteration, t);
    else
      saveSolution_aux<IO_UNIQUE>(Q, iteration, t);
  }

  template<enum SaveMethod save_method>
  void saveSolution_aux(const Array &Q, int iteration, real_t t)
  {
    constexpr bool is_multiple = (save_method == IO_MULTIPLE);
    std::ostringstream oss;
    
    std::string iteration_str, basename;
    if constexpr (is_multiple) {
      oss << params.filename_out << "_" << std::setw(ite_nzeros) << std::setfill('0') << iteration;
      iteration_str = oss.str();
      basename = oss.str();
    }
    else {
      oss << "ite_" << std::setw(ite_nzeros) << std::setfill('0') << iteration;
      iteration_str = oss.str();
      basename = params.filename_out;
    }
    
    const std::string h5_filename  = basename + ".h5";
    const std::string xmf_filename = basename + ".xmf";

    const bool write_attrs = (is_multiple || first_iteration);
      
    const auto flag_h5 = (write_attrs ? File::Truncate : File::ReadWrite);
    File file(params.output_path + h5_filename, flag_h5);

    if (write_attrs) {
      first_iteration = false;
      file.createAttribute("Ntx", device_params.Ntx);
      file.createAttribute("Nty", device_params.Nty);
      file.createAttribute("Nx", device_params.Nx);
      file.createAttribute("Ny", device_params.Ny);
      file.createAttribute("ibeg", device_params.ibeg);
      file.createAttribute("iend", device_params.iend);
      file.createAttribute("jbeg", device_params.jbeg);
      file.createAttribute("jend", device_params.jend);
      file.createAttribute("problem", params.problem);

      Table<2> coordinates;
      coordinates.reserve((device_params.Nx + 1) * (device_params.Ny + 1));
      // -- vertex pos
      for (int j=device_params.jbeg; j <= device_params.jend; ++j) {
        for (int i=device_params.ibeg; i <= device_params.iend; ++i) {
          coordinates.push_back({
            (i-device_params.ibeg) * device_params.dx + device_params.xmin, 
            (j-device_params.jbeg) * device_params.dy + device_params.ymin 
          });
        }
      }
      file.createDataSet("coordinates", coordinates);

      std::vector<std::array<uint32_t, 4>> connectivity;
      connectivity.reserve(device_params.Nx * device_params.Ny);
      // -- connectivity
      for (int j=device_params.jbeg; j < device_params.jend; ++j) {
        for (int i=device_params.ibeg; i < device_params.iend; ++i) {
          auto vertex_id = [&](int i, int j) -> uint32_t { return (i-device_params.ibeg) + (j-device_params.jbeg) * (device_params.Nx + 1); };
          connectivity.push_back({vertex_id(i, j), vertex_id(i+1, j), vertex_id(i+1, j+1), vertex_id(i, j+1)});
        }
      }
      file.createDataSet("connectivity", connectivity);
    }
    
    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    auto tvars = make_tables();
    for (int j=device_params.jbeg; j<device_params.jend; ++j) {
      for (int i=device_params.ibeg; i<device_params.iend; ++i) {
        for_each_tuples([&](const auto &io_var, auto& tvar) {
          constexpr std::size_t n_var = std::tuple_size_v<typename std::remove_reference_t<decltype(io_var)>::second_type>;
          
          tvar.emplace_back();
          auto &back = tvar.back();
          for (std::size_t id=0; id<n_var; id++)
            back[id] = Qhost(j, i, io_var.second[id]);
        }, io_variables, tvars);
      }
    }

    auto save_groups = [&](auto &g) {
      for_each_tuples([&](const auto &io_var, const auto& tvar) {
        const std::string var_name{io_var.first};
        g.createDataSet(var_name, tvar);
      }, io_variables, tvars);

      g.createAttribute("time", t);
      g.createAttribute("iteration", iteration);
    };

    if constexpr (is_multiple) {
      save_groups(file);
    }
    else {
      auto h5_group = file.createGroup(iteration_str);
      save_groups(h5_group);
    }

    std::string group;
    if constexpr (is_multiple) {
      group = "";
    }
    else {
      group = iteration_str + "/";
    }

    constexpr auto flag_xdmf = std::fstream::out | (is_multiple ? std::fstream::trunc : std::fstream::in);
    std::fstream xdmf_file(params.output_path + xmf_filename, flag_xdmf);
    if constexpr (!is_multiple) {
      xdmf_file.seekp(-str_xdmf_footer_unique.size(), std::fstream::end);
    }

    const auto xdmf_print_format = get_xdmf_format_tuple(iteration_str.c_str(), t, h5_filename.c_str(), group.c_str());
    xdmf_file << format_string(xdmf_print_format);

    // xdmf main for multiple output
    if constexpr (is_multiple) {
      std::fstream xdmf_main_file(params.output_path + params.filename_out + "_main.xmf", std::fstream::out | std::fstream::in);
      xdmf_main_file.seekp(-str_xdmf_main_footer.size(), std::fstream::end);
      xdmf_main_file << format_string(R"xml(
    <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)" />)xml", xmf_filename.c_str()) << str_xdmf_main_footer.data();
    }
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

    if ( !params.multiple_outputs && std::filesystem::equivalent(restart_file, params.output_path + params.filename_out + ".h5") ) {
      if (delim_multi != std::string::npos) {
        std::cerr << "Invalid restart file : if your restart file and output file are "
                     "the same, you can only start from the last iteration." << std::endl << std::endl;
        throw std::runtime_error("ERROR : Invalid restart_file.");
      }
      // do not truncate the file if restart_file is the same as output_file.
      this->first_iteration = false;
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
        const size_t last_ite_index = file.getNumberObjects() - 1; // first groups are : coordinates/ and connectivity/, following by all the ite_xxxx/. 
        group = file.getObjectName(last_ite_index);
      }
      HighFive::Group h5_group = file.getGroup(group);
      HighFive::Attribute attr_time = h5_group.getAttribute("time");
      attr_time.read(time);
      HighFive::Attribute attr_ite = h5_group.getAttribute("iteration");
      attr_ite.read(iteration);
      group = group + "/";
    }

    auto Nt = getShape(file, group + std::string{std::get<0>(io_variables).first})[0];

    if (Nt != device_params.Nx*device_params.Ny) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << device_params.Nx << "x" << device_params.Ny << "=" << device_params.Nx*device_params.Ny << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);

    std::cout << "Loading restart data from hdf5" << std::endl;

    auto tvars = make_tables();
    for_each_tuples([&](const auto &io_var, auto &tvar) {
      const std::string var_name{io_var.first};
      tvar = std::move(load<std::remove_reference_t<decltype(tvar)>>(file, group + var_name));
    }, io_variables, tvars);

    std::size_t lid = 0;
    for (int y=0; y < device_params.Ny; ++y) {
      for (int x=0; x < device_params.Nx; ++x) {
        for_each_tuples([&](const auto &io_var, auto& tvar) {
          constexpr std::size_t n_var = std::tuple_size_v<typename std::remove_reference_t<decltype(io_var)>::second_type>;

          for (std::size_t id=0; id < n_var; id++)
            Qhost(y+device_params.jbeg, x+device_params.ibeg, io_var.second[id]) = tvar[lid][id];
        }, io_variables, tvars);
        lid++;
      }
    }

    Kokkos::deep_copy(Q, Qhost);

    BoundaryManager bc(params);
    bc.fillBoundaries(Q);

    if (time + params.device_params.epsilon > params.tend) {
      std::cerr << "Restart time is greater than end time : " << std::endl
                << "  time: " << time << "\ttend: " << params.tend << std::endl << std::endl; 
      throw std::runtime_error("ERROR : restart time is greater than the end time.");
    }

    std::cout << "Restart finished !" << std::endl;

    if (first_iteration) {
      file.~File(); // free the h5 before saving
      saveSolution(Q, iteration, time);
    }

    return {time, iteration};
  }
};
}
cmake_minimum_required(VERSION 3.16)

project(fv2d LANGUAGES C CXX)

# Various sets
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/kokkos)
add_subdirectory(external/HighFive)
include_directories(external/inih)


add_executable(fv2d
  main.cpp)

target_link_libraries(fv2d PRIVATE HighFive Kokkos::kokkos)

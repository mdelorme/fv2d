cmake_minimum_required(VERSION 3.16)

project(fv2d LANGUAGES C CXX)

find_package(OpenMP)

add_subdirectory(external/HighFive)

include_directories(external/inih)

# Various sets
set(CMAKE_CXX_STANDARD 17)

add_executable(fv2d
  main.cpp)

if (OpenMP_CXX_FOUND)
  target_link_libraries(fv2d PRIVATE HighFive OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenM not found, compiling in serial")
  target_link_libraries(fv2d PRIVATE HighFive)
endif()
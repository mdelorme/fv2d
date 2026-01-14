#pragma once

#include <Kokkos_Random.hpp>
#include "../SimInfo.h"

using RandomPool = Kokkos::Random_XorShift64_Pool<>;

namespace fv2d
{

struct InitData
{
  RandomPool random_pool;
  InitData() = default;
  InitData(const Params &full_params) : random_pool(full_params.seed) {};
  ~InitData() = default;
};

} // namespace fv2d

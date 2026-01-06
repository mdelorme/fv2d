#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <Kokkos_Random.hpp>
#include "../SimInfo.h"

using RandomPool = Kokkos::Random_XorShift64_Pool<>;

namespace fv2d
{

struct InitData
{
  RandomPool random_pool;
  InitData() = default;
  InitData(Params &full_params) : random_pool(full_params.seed) {};
  ~InitData() = default;
};

using InitFunction = std::function<void(Array, int, int, const DeviceParams &, const InitData &)>;

inline std::unordered_map<std::string, InitFunction> registry;

class InitRegistry
{
public:
  static void registerFunction(const std::string &name, InitFunction func) { registry[name] = func; }

  static InitFunction getFunction(const std::string &name)
  {
    if (registry.find(name) == registry.end())
    {
      throw std::runtime_error("Unknown problem: " + name);
    }
    return registry[name];
  }
};

#define REGISTER_PROBLEM(func, name)                                                                                   \
  namespace                                                                                                            \
  {                                                                                                                    \
  struct Register##func                                                                                                \
  {                                                                                                                    \
    Register##func() { InitRegistry::registerFunction(name, &func); }                                                  \
  } register_##func;                                                                                                   \
  }

} // namespace fv2d

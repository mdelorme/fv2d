#pragma once

#include <unordered_map>

#include "InitData.h"
#include "../SimInfo.h"

namespace fv2d {

/** 
 * @brief Abstract class used for initialization of the domain
 */
struct InitFormula {
  virtual void init(Array Q, const Params &full_params) = 0;
};

} // namespace fv2d
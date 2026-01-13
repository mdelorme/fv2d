#pragma once

#include "InitFormula.h"

// Macro to register an initialization type
#define REGISTER_INIT(class_type, id_name) \
namespace { \
  struct Register##id_name { \
    Register##id_name() { InitFactory::registerFormula<class_type>(#id_name); } \
  } register##id_name; \
} 


namespace fv2d {

/**
 * @brief Factory registering init types
 */
struct InitFactory {
  static std::map<std::string, std::shared_ptr<InitFormula> > formulae;

  /**
   * @brief stores a reference to an init formula as a string
   * @tparam Formula the type of the initialization functor. Must inherit InitFormula
   * @param formula_name name of the formula to store in the map
   */
  template <typename Formula>
  static bool registerFormula(std::string formula_name) {
    if (formulae.count(formula_name) != 0) {
      std::cerr << "InitFormula " << formula_name << " already registered" << std::endl;
      return false;
    } 
    formulae[formula_name] = std::make_shared<Formula>();
    return true;
  }

  /**
   * @brief Instantiate a formula corresponding to a given name
   * @param formula_name the name of the formula to instantiate
   */
  static std::shared_ptr<InitFormula> instantiate(std::string formula_name) {
    if (formulae.count(formula_name) == 0) {
      std::cerr << "Cannot find problem " << formula_name << std::endl;
      std::cerr << "Available problems : " << std::endl;
      for (auto &f : formulae) 
        std::cerr << "  " << f.first << std::endl;
      return nullptr;
    }
    
    return formulae[formula_name];
  }
};

std::map<std::string, std::shared_ptr<InitFormula> > InitFactory::formulae;

} // namespace fv2d
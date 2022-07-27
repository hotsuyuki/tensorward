#pragma once

#include <vector>

#include "tensorward/core/function.h"
#include "tensorward/core/parameter.h"

namespace tensorward::core {

class Optimizer {
 public:
  Optimizer() {}

  virtual ~Optimizer() {}

  void Update(const std::vector<ParameterSharedPtr>& param_ptrs);

  virtual void UpdateSingleParameter(const ParameterSharedPtr param_ptr) = 0;

 protected:
  // TODO: Add something like `preprocess_functions_`.
};
  
}  // namespace tensorward::core

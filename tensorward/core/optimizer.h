#pragma once

#include <vector>

#include "tensorward/core/function.h"
#include "tensorward/core/parameter.h"

namespace tensorward::core {

class Optimizer {
 public:
  Optimizer() {}

  virtual ~Optimizer() {}

  void Update(const std::vector<ParameterSharedPtr>& param_ptrs) const;

  virtual void UpdateSingleParameter(const ParameterSharedPtr param_ptr) const = 0;

 protected:
  // TODO: Add something like `preprocess_function_lambdas_`.
};
  
}  // namespace tensorward::core

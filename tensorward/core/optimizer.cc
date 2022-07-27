#include "tensorward/core/optimizer.h"

#include <algorithm>
#include <iterator>

namespace tensorward::core {

void Optimizer::Update(const std::vector<ParameterSharedPtr>& param_ptrs) {
  std::vector<ParameterSharedPtr> valid_param_ptrs;
  std::copy_if(param_ptrs.begin(), param_ptrs.end(), std::back_inserter(valid_param_ptrs),
               [](const ParameterSharedPtr param_ptr) { return param_ptr->grad_opt().has_value(); });

  // TODO: Implement a for-loop for preprocessing of the parameters using `preprocess_functions_`.

  for (const auto& valid_param_ptr : valid_param_ptrs) {
    UpdateSingleParameter(valid_param_ptr);
  }
}

}  // namespace tensorward::core

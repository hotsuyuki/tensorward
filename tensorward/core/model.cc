#include "tensorward/core/model.h"

namespace tensorward::core {

void Model::ClearGrads() {
  for (const auto& layer_ptr : layer_ptrs_) {
    layer_ptr->ClearGrads();
  }
}

const std::vector<ParameterSharedPtr> Model::GetParamPtrs() const {
  std::size_t num_param_ptrs = 0;
  for (const auto& layer_ptr : layer_ptrs_) {
    num_param_ptrs += layer_ptr->param_map().size();
  }

  std::vector<ParameterSharedPtr> param_ptrs;
  param_ptrs.reserve(num_param_ptrs);
  for (const auto& layer_ptr : layer_ptrs_) {
    for (const auto& param_name_ptr : layer_ptr->param_map()) {
      const ParameterSharedPtr param_ptr = param_name_ptr.second;
      param_ptrs.push_back(param_ptr);
    }
  }

  return param_ptrs;
}

}  // namespace tensorward::core

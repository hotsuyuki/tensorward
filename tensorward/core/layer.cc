#include "tensorward/core/layer.h"

namespace tensorward::core {

const std::vector<TensorSharedPtr> Layer::Call(const std::vector<TensorSharedPtr>& input_tensor_ptrs) {
  const std::vector<TensorSharedPtr> output_tensor_ptrs = Forward(input_tensor_ptrs);

  input_tensor_ptrs_.clear();
  input_tensor_ptrs_.reserve(input_tensor_ptrs.size());
  for (const auto& input_tensor_ptr : input_tensor_ptrs) {
    // Converts from "shared" pointers to "weak" pointers.
    input_tensor_ptrs_.push_back(input_tensor_ptr);
  }

  output_tensor_ptrs_.clear();
  output_tensor_ptrs_.reserve(output_tensor_ptrs.size());
  for (const auto& output_tensor_ptr : output_tensor_ptrs) {
    // Converts from "shared" pointers to "weak" pointers.
    output_tensor_ptrs_.push_back(output_tensor_ptr);
  }

  return output_tensor_ptrs;
}

void Layer::ClearGrads() {
  for (const auto& param_name_ptr : param_map_) {
    const ParameterSharedPtr param_ptr = param_name_ptr.second;
    param_ptr->ClearGrad();
  }
}

}  // namespace tensorward::core

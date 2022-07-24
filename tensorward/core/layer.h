#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorward/core/parameter.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Layer {
 public:
  Layer() {}

  virtual ~Layer() {}

  // Performs the forward calculation and the computational graph growth (in the function of this layer).
  const std::vector<TensorSharedPtr> Call(const std::vector<TensorSharedPtr>& input_tensor_ptrs);

  // Performs the forward calculation of the function of this layer.
  virtual const std::vector<TensorSharedPtr> Forward(const std::vector<TensorSharedPtr>& input_tensor_ptrs) = 0;

  void ClearGrads();

  const std::unordered_map<std::string, ParameterSharedPtr>& param_map() const { return param_map_; }

  const std::vector<TensorWeakPtr>& input_tensor_ptrs() const { return input_tensor_ptrs_; }

  const std::vector<TensorWeakPtr>& output_tensor_ptrs() const { return output_tensor_ptrs_; }

 protected:
  std::unordered_map<std::string, ParameterSharedPtr> param_map_;

  std::vector<TensorWeakPtr> input_tensor_ptrs_;

  std::vector<TensorWeakPtr> output_tensor_ptrs_;
};

using LayerSharedPtr = std::shared_ptr<Layer>;

}  // namespace tensorward::core

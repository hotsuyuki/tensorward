#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "tensorward/core/function.h"
#include "tensorward/core/layer.h"
#include "tensorward/core/model.h"
#include "tensorward/core/tensor.h"
#include "tensorward/layer/linear.h"

namespace tensorward::model {

class MultiLayerPerceptron : public core::Model {
 public:
  MultiLayerPerceptron(const std::vector<std::size_t>& out_sizes,
                       const core::FunctionLambda& activation_function_lambda)
      : out_sizes_(out_sizes), activation_function_lambda_(activation_function_lambda) {
    for (const auto& out_size : out_sizes) {
      const core::LayerSharedPtr layer_ptr = std::make_shared<layer::Linear>(out_size); 
      layer_ptrs_.push_back(layer_ptr);
    }
  }

  ~MultiLayerPerceptron() {}

  const std::vector<core::TensorSharedPtr> Predict(
      const std::vector<core::TensorSharedPtr>& input_tensor_ptrs) override {
    std::vector<core::TensorSharedPtr> output_tensor_ptrs(input_tensor_ptrs);
    for (std::size_t i = 0; i < layer_ptrs_.size() - 1; ++i) {
      output_tensor_ptrs = layer_ptrs_[i]->Call(output_tensor_ptrs);
      output_tensor_ptrs = activation_function_lambda_(output_tensor_ptrs);
    }
    output_tensor_ptrs = layer_ptrs_[layer_ptrs_.size() - 1]->Call(output_tensor_ptrs);

    return output_tensor_ptrs;
  }

  const std::vector<std::size_t>& out_sizes() { return out_sizes_; }

  const core::FunctionLambda& activation_function_lambda() { return activation_function_lambda_; }

 private:
  std::vector<std::size_t> out_sizes_;

  core::FunctionLambda activation_function_lambda_;
};

}  // namespace tensorward::model

#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Sigmoid : public core::Function {
 public:
  Sigmoid() : core::Function({.num_inputs = 1, .num_outputs = 1}) {}

  ~Sigmoid() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = 1 / (1 + exp(-x))
    const xt::xarray<float> y = 1.0 / (1.0 + xt::exp(-xs[0]));

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& y = output_tensor_ptrs_[0].lock()->data();

    // y = 1 / (1 + exp(-x)) ---> dy_dx = y * (1 - y) ---> dL_dx = dL_dy * dy_dx = dL_dy * y * (1 - y)
    const xt::xarray<float> dy_dx = y * (1.0 - y);
    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    return {dL_dx};
  }
};

const core::TensorSharedPtr sigmoid(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr sigmoid_function_ptr = std::make_shared<Sigmoid>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = sigmoid_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

const core::FunctionLambda sigmoid_lambda = [](const std::vector<core::TensorSharedPtr>& input_tensor_ptrs) {
  const core::TensorSharedPtr output_tensor_ptr = sigmoid(input_tensor_ptrs[0]);
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs({output_tensor_ptr});

  return output_tensor_ptrs;
};

}  // namespace tensorward::function

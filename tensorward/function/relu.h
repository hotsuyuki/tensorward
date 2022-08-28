#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class ReLU : public core::Function {
 public:
  ReLU() : core::Function({.num_inputs = 1, .num_outputs = 1}) {}

  ~ReLU() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = x (if 0 < x), y = 0 (if x <= 0) ---> y = max(0, x)
    const xt::xarray<float> y = xt::maximum(xt::zeros_like(xs[0]), xs[0]);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();

    // y = x (if 0 < x), y = 0 (if x <= 0) ---> dy_dx = 1 (if 0 < x), dy_dx = 0 (if x <= 0) ---> dy_dx is like a mask.
    const xt::xarray<float> dy_dx = (0.0 < x);
    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    return {dL_dx};
  }
};

const core::TensorSharedPtr relu(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr relu_function_ptr = std::make_shared<ReLU>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = relu_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

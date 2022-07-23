#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Square : public core::Function {
 public:
  Square() : core::Function({.num_inputs = 1, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = x^2
    const xt::xarray<float> y = xt::square(xs[0]);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();

    // y = x^2 ---> dy_dx = 2x ---> dL_dx = dL_dy * dy_dx = dL_dy * 2x
    const xt::xarray<float> dy_dx = 2 * x;
    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    return {dL_dx};
  }
};

const core::TensorSharedPtr square(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr square_function_ptr = std::make_shared<Square>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = square_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

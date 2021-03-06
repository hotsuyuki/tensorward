#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Exp : public core::Function {
 public:
  Exp() : core::Function({.num_inputs = 1, .num_outputs = 1}) {}

  ~Exp() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = exp(x)
    const xt::xarray<float> y = xt::exp(xs[0]);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& y = output_tensor_ptrs_[0].lock()->data();

    // y = exp(x) ---> dy_dx = exp(x) = y ---> dL_dx = dL_dy * dy_dx = dL_dy * exp(x) = dL_dy * y
    const xt::xarray<float>& dy_dx = y;
    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    return {dL_dx};
  }
};

const core::TensorSharedPtr exp(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr exp_function_ptr = std::make_shared<Exp>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = exp_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

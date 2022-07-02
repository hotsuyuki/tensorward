#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Div : public Function {
 public:
  Div() : Function({.num_inputs = 2, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) const override {
    // y = x0 / x1
    const xt::xarray<float> y = xs[0] / xs[1];

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) const override {
    const xt::xarray<float>& x0 = input_tensor_ptrs_[0]->data();
    const xt::xarray<float>& x1 = input_tensor_ptrs_[1]->data();

    // y = x0 / x1 = x1^(-1) * x0 ---> dy_dx0 = x1^(-1) ---> dL_dx0 = dL_dy * dy_dx0 = dL_dy * 1/x1
    const xt::xarray<float> dy_dx0 = 1.0 / x1;
    const xt::xarray<float> dL_dx0 = dL_dys[0] * dy_dx0;

    // y = x0 / x1 = x0 * x1^(-1) ---> dy_dx1 = -x0 * x1^(-2) ---> dL_dx1 = dL_dy * dy_dx1 = dL_dy * (-x0/(x1)^2)
    const xt::xarray<float> dy_dx1 = -x0 / xt::square(x1);
    const xt::xarray<float> dL_dx1 = dL_dys[0] * dy_dx1;

    return {dL_dx0, dL_dx1};
  }
};

const TensorSharedPtr div(const TensorSharedPtr input_tensor_ptr0, const TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const FunctionSharedPtr div_function_ptr = std::make_shared<Div>();
  const std::vector<TensorSharedPtr> output_tensor_ptrs =
      div_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});

  return output_tensor_ptrs[0];
}

const TensorSharedPtr operator/(const TensorSharedPtr lhs_ptr, const TensorSharedPtr rhs_ptr) {
  return div(lhs_ptr, rhs_ptr);
}

const TensorSharedPtr operator/(const TensorSharedPtr lhs_ptr, const xt::xarray<float>& rhs) {
  return lhs_ptr / AsTensorSharedPtr(rhs);
}

const TensorSharedPtr operator/(const xt::xarray<float>& lhs, const TensorSharedPtr rhs_ptr) {
  return AsTensorSharedPtr(lhs) / rhs_ptr;
}

} // namespace tensorward::core

#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Sub : public Function {
 public:
  Sub() : Function({.num_inputs = 2, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) const override {
    // y = x0 - x1
    const xt::xarray<float> y = xs[0] - xs[1];

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) const override {
    // y = x0 - x1 ---> dy_dx0 = 1 ---> dL_dx0 = dL_dy * dy_dx0 = dL_dy * 1 = dL_dy
    const xt::xarray<float>& dL_dx0 = dL_dys[0];

    // y = x0 - x1 ---> dy_dx1 = -1 ---> dL_dx1 = dL_dy * dy_dx1 = dL_dy * (-1) = -dL_dy
    const xt::xarray<float> dL_dx1 = -dL_dys[0];

    return {dL_dx0, dL_dx1};
  }
};

const TensorSharedPtr sub(const TensorSharedPtr input_tensor_ptr0, const TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const FunctionSharedPtr sub_function_ptr = std::make_shared<Sub>();
  const std::vector<TensorSharedPtr> output_tensor_ptrs =
      sub_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});

  return output_tensor_ptrs[0];
}

const TensorSharedPtr operator-(const TensorSharedPtr lhs_ptr, const TensorSharedPtr rhs_ptr) {
  return sub(lhs_ptr, rhs_ptr);
}

const TensorSharedPtr operator-(const TensorSharedPtr lhs_ptr, const xt::xarray<float>& rhs) {
  return lhs_ptr - AsTensorSharedPtr(rhs);
}

const TensorSharedPtr operator-(const xt::xarray<float>& lhs, const TensorSharedPtr rhs_ptr) {
  return AsTensorSharedPtr(lhs) - rhs_ptr;
}

const TensorSharedPtr operator-(const TensorSharedPtr lhs_ptr, const float rhs) {
  return lhs_ptr - AsTensorSharedPtr(rhs);
}

const TensorSharedPtr operator-(const float lhs, const TensorSharedPtr rhs_ptr) {
  return AsTensorSharedPtr(lhs) - rhs_ptr;
}

} // namespace tensorward::core

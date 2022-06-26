#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Neg : public Function {
 public:
  Neg() : Function({.num_inputs = 1, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) const override {
    // y = -x
    const xt::xarray<float> y = -xs[0];

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) const override {
    // y = -x ---> dy_dx = -1 ---> dL_dx = dL_dy * dy_dx = dL_dy * (-1) = -dL_dy
    const xt::xarray<float> dL_dx = -dL_dys[0];

    return {dL_dx};
  }
};

const TensorSharedPtr neg(const TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const FunctionSharedPtr neg_function_ptr = std::make_shared<Neg>();
  const std::vector<TensorSharedPtr> output_tensor_ptrs = neg_function_ptr->Call(input_tensor_ptr);

  return output_tensor_ptrs[0];
}

const TensorSharedPtr operator-(const TensorSharedPtr lhs_ptr) {
  return neg(lhs_ptr);
}

} // namespace tensorward::core

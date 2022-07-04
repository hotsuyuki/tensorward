#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Pow : public core::Function {
 public:
  Pow(const int exponent) : core::Function({.num_inputs = 1, .num_outputs = 1}), exponent_(exponent) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = x^e
    const xt::xarray<float> y = xt::pow(xs[0], exponent_);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();

    // y = x^e ---> dy_dx = e * x^(e - 1) ---> dL_dx = dL_dy * dy_dx = dL_dy * (e * x^(e - 1))
    const xt::xarray<float> dy_dx = static_cast<float>(exponent_) * xt::pow(x, exponent_ - 1);
    const xt::xarray<float> dL_dx = dL_dys[0] * dy_dx;

    return {dL_dx};
  }

  const int exponent() const { return exponent_; }

 private:
  int exponent_;
};

const core::TensorSharedPtr pow(const core::TensorSharedPtr input_tensor_ptr, const int exponent) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr pow_function_ptr = std::make_shared<Pow>(exponent);
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = pow_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

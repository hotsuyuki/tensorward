#pragma once

#include <memory>

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Exp : public core::Function {
 public:
  const xt::xarray<float> Forward(const xt::xarray<float>& x) const override {
    const xt::xarray<float> y = xt::exp(x);

    return y;
  }

  const xt::xarray<float> Backward(const xt::xarray<float>& dL_dy) const override {
    const xt::xarray<float>& dy_dx = output_tensor_ptr_.lock()->data();
    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    return dL_dx;
  }
};

const core::TensorSharedPtr exp(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr exp_function_ptr = std::make_shared<Exp>();
  const core::TensorSharedPtr output_tensor_ptr = exp_function_ptr->Call(input_tensor_ptr);

  return output_tensor_ptr;
}

}  // namespace tensorward::function

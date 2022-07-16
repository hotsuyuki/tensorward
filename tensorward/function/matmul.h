#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Matmul : public core::Function {
 public:
  Matmul() : core::Function({.num_inputs = 2, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    const xt::xarray<float>& x = xs[0];
    const xt::xarray<float>& W = xs[1];

    // y = x W
    const xt::xarray<float> y = xt::linalg::dot(x, W);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();
    const xt::xarray<float>& W = input_tensor_ptrs_[1]->data();

    // y = x W ---> dL_dx = dL_dy W.T
    const xt::xarray<float> dL_dx = xt::linalg::dot(dL_dys[0], xt::transpose(W));

    // y = x W ---> dL_dW = x.T dL_dy
    const xt::xarray<float> dL_dW = xt::linalg::dot(xt::transpose(x), dL_dys[0]);

    return {dL_dx, dL_dW};
  }
};

const core::TensorSharedPtr matmul(const core::TensorSharedPtr input_tensor_ptr0,
                                   const core::TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr matmul_function_ptr = std::make_shared<Matmul>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs =
      matmul_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});
  
  return output_tensor_ptrs[0];
}

} // namespace tensorward::function

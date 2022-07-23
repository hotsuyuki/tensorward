#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"
#include "tensorward/util/xtensor_sum_to.h"

namespace tensorward::function {

class Linear : public core::Function {
 public:
  Linear() : core::Function({.num_inputs = 3, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    const xt::xarray<float>& x = xs[0];
    const xt::xarray<float>& W = xs[1];
    const xt::xarray<float>& b = xs[2];

    // y = x W + b
    const xt::xarray<float> y = xt::linalg::dot(x, W) + b;

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();
    const xt::xarray<float>& W = input_tensor_ptrs_[1]->data();
    const xt::xarray<float>& b = input_tensor_ptrs_[2]->data();

    // y = x W + b ---> dL_dx = dL_dy W.T
    const xt::xarray<float> dL_dx = xt::linalg::dot(dL_dy, xt::transpose(W));

    // y = x W + b ---> dL_dW = x.T dL_dy
    const xt::xarray<float> dL_dW = xt::linalg::dot(xt::transpose(x), dL_dy);

    // y = x W + b ---> dy_db = 1 ---> dL_db = dL_dy * dy_db = dL_dy * 1 = dL_dy
    xt::xarray<float> dL_db = dL_dy;

    // Reduces the shape of dL_db if b was broadcasted during the forward calculation.
    const bool is_broadcasted_for_b = (dL_dy.shape() != b.shape());
    if (is_broadcasted_for_b) {
      dL_db = util::XtensorSumTo(dL_db, b.shape());
    }

    return {dL_dx, dL_dW, dL_db};
  }
};

const core::TensorSharedPtr linear(const core::TensorSharedPtr input_tensor_ptr0,
                                   const core::TensorSharedPtr input_tensor_ptr1,
                                   const core::TensorSharedPtr input_tensor_ptr2) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr linear_function_ptr = std::make_shared<Linear>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs =
      linear_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1, input_tensor_ptr2});
  
  return output_tensor_ptrs[0];
}

} // namespace tensorward::function

#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"
#include "tensorward/util/xtensor_sum_to.h"

namespace tensorward::core {

class Add : public Function {
 public:
  Add() : Function({.num_inputs = 2, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // y = x0 + x1
    const xt::xarray<float> y = xs[0] + xs[1];

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    // y = x0 + x1 ---> dy_dx0 = 1 ---> dL_dx0 = dL_dy * dy_dx0 = dL_dy * 1 = dL_dy
    xt::xarray<float> dL_dx0 = dL_dys[0];

    // y = x0 + x1 ---> dy_dx1 = 1 ---> dL_dx1 = dL_dy * dy_dx1 = dL_dy * 1 = dL_dy
    xt::xarray<float> dL_dx1 = dL_dys[0];

    // Reduces the shape of dL_dx0 or dL_dx1 if either x0 or x1 was broadcasted during the forward calculation.
    const xt::xarray<float>::shape_type& x0_shape = input_tensor_ptrs_[0]->data().shape();
    const xt::xarray<float>::shape_type& x1_shape = input_tensor_ptrs_[1]->data().shape();
    const bool is_broadcasted_for_x0_or_x1 = (x0_shape != x1_shape);
    if (is_broadcasted_for_x0_or_x1) {
      dL_dx0 = util::XtensorSumTo(dL_dx0, x0_shape);
      dL_dx1 = util::XtensorSumTo(dL_dx1, x1_shape);
    }

    return {dL_dx0, dL_dx1};
  }
};

const TensorSharedPtr add(const TensorSharedPtr input_tensor_ptr0, const TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const FunctionSharedPtr add_function_ptr = std::make_shared<Add>();
  const std::vector<TensorSharedPtr> output_tensor_ptrs =
      add_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});
  
  return output_tensor_ptrs[0];
}

const TensorSharedPtr operator+(const TensorSharedPtr lhs_ptr, const TensorSharedPtr rhs_ptr) {
  return add(lhs_ptr, rhs_ptr);
}

const TensorSharedPtr operator+(const TensorSharedPtr lhs_ptr, const xt::xarray<float>& rhs) {
  return lhs_ptr + AsTensorSharedPtr(rhs);
}

const TensorSharedPtr operator+(const xt::xarray<float>& lhs, const TensorSharedPtr rhs_ptr) {
  return AsTensorSharedPtr(lhs) + rhs_ptr;
}

} // namespace tensorward::core

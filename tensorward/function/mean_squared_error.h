#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class MeanSquaredError : public core::Function {
 public:
  MeanSquaredError() : core::Function({.num_inputs = 2, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    assert(xs[0].shape() == xs[1].shape());
    const std::size_t& num_data = xs[0].shape(0);

    // y = sum((x0 - x1)^2) / N
    const xt::xarray<float> y = xt::sum(xt::square(xs[0] - xs[1])) / num_data;

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& x0 = input_tensor_ptrs_[0]->data();
    const xt::xarray<float>& x1 = input_tensor_ptrs_[1]->data();

    assert(x0.shape() == x1.shape());
    const std::size_t& num_data = x0.shape(0);

    // Suppose we introduce an intermidiate function `a(x0) = (x0 - x1)^2 / N`, then we can re-write `y(x0)` to `y(a)`,
    //
    //   y(a) = sum((x0 - x1)^2) / N = sum((x0 - x1)^2 / N) = sum(a)
    //
    // The backward calculation of `sum()` is `broadcast_to()`, so `dL_da` is `dL_da = broadcast_to(dL_dy, a_shape)`.
    // And `da_dx0` is `da_dx0 = 2(x0 - x1) / N`, so eventually `dL_dx0` would be
    //
    //   dL_dx0 = dL_da * da_dx0 = broadcast_to(dL_dy, a_shape) * (2(x0 - x1) / N)
    //
    const xt::xarray<float> dL_dx0 = xt::broadcast(dL_dy, x0.shape()) * (2.0 * (x0 - x1) / num_data);

    // Suppose we introduce an intermidiate function `b(x1) = (x0 - x1)^2 / N`, then we can re-write `y(x1)` to `y(b)`,
    //
    //   y(b) = sum((x0 - x1)^2) / N = sum((x0 - x1)^2 / N) = sum(b)
    //
    // The backward calculation of `sum()` is `broadcast_to()`, so `dL_db` is `dL_db = broadcast_to(dL_dy, b_shape)`.
    // And `db_dx1` is `db_dx1 = -2(x0 - x1) / N`, so eventually `dL_dx1` would be
    //
    //   dL_dx1 = dL_db * db_dx1 = broadcast_to(dL_dy, b_shape) * (-2(x0 - x1) / N)
    //
    // which is equal to `-dL_dx0`. (Note that `x0_shape = a_shape = b_shape = x1_shape`)
    //
    const xt::xarray<float> dL_dx1 = -dL_dx0;

    return {dL_dx0, dL_dx1};
  }
};

const core::TensorSharedPtr mean_squared_error(const core::TensorSharedPtr input_tensor_ptr0,
                                               const core::TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr mean_squared_error_function_ptr = std::make_shared<MeanSquaredError>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs =
      mean_squared_error_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});
  
  return output_tensor_ptrs[0];
}

} // namespace tensorward::function

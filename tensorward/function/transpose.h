#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Transpose : public core::Function {
 public:
  // TODO: Extend `Transpose` class to accept optional `permutation` as an input argument.
  // TODO: e.g. `Transpose(const xt::xarray<float>::shape_type& permutation = None) : ..., permutation_(permutation) {}`
  Transpose() : core::Function({.num_inputs = 1, .num_outputs = 1}) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    const xt::xarray<float> y = xt::transpose(xs[0]);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float> dL_dx = xt::transpose(dL_dys[0]);

    return {dL_dx};
  }
};

// TODO: Extend `transpose()` function to accept optional `permutation` as an input argument.
// TODO: e.g. `transpose(..., xt::xarray<float>::shape_type& permutation = None) { ... }`
const core::TensorSharedPtr transpose(const core::TensorSharedPtr input_tensor_ptr) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr transpose_function_ptr = std::make_shared<Transpose>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = transpose_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

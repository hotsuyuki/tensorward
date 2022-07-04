#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Reshape : public core::Function {
 public:
  Reshape(const xt::xarray<float>::shape_type& output_shape)
      : core::Function({.num_inputs = 1, .num_outputs = 1}), output_shape_(output_shape) {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    input_shape_ = xs[0].shape();

    xt::xarray<float> y = xs[0];
    y.reshape(output_shape_);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    xt::xarray<float> dL_dx = dL_dys[0];
    dL_dx.reshape(input_shape_);

    return {dL_dx};
  }

  const xt::xarray<float>::shape_type& input_shape() const { return input_shape_; }

  const xt::xarray<float>::shape_type& output_shape() const { return output_shape_; }

 private:
  xt::xarray<float>::shape_type input_shape_;

  xt::xarray<float>::shape_type output_shape_;
};

const core::TensorSharedPtr reshape(const core::TensorSharedPtr input_tensor_ptr,
                                    const xt::xarray<float>::shape_type& output_shape) {
  // If the input shape is the same as the output shape, then no need to reshape.
  if (input_tensor_ptr->data().shape() == output_shape) {
    return input_tensor_ptr;
  }

  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr reshape_function_ptr = std::make_shared<Reshape>(output_shape);
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = reshape_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

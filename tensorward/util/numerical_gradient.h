#pragma once

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"

namespace tensorward::util {

const xt::xarray<float> NumericalGradient(
    const core::FunctionSharedPtr function_ptr, const xt::xarray<float>& input_data, const float epsilon = 1.0e-3) {
  // Central finite difference.
  const xt::xarray<float> output_data_positive = function_ptr->Forward(input_data + epsilon);
  const xt::xarray<float> output_data_negative = function_ptr->Forward(input_data - epsilon);
  const xt::xarray<float> numerical_grad = (output_data_positive - output_data_negative) / (2.0 * epsilon);

  return numerical_grad;
}

}  // namespace tensorward::util

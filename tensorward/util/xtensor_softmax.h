#pragma once

#include <xtensor/xarray.hpp>

namespace tensorward::util {

const xt::xarray<float> XtensorSoftmax(const xt::xarray<float>& input_data) {
  // Subtracts the max element along with the last axis from the input data in order to avoid overflow when `exp()`.
  const xt::xarray<float> max_input_data = xt::amax(input_data, {-1}, xt::keep_dims);
  const xt::xarray<float> modified_input_data = input_data - max_input_data;

  // NOTE: Need to construct temporary variables explicitly for the intermidiate results,
  // NOTE: otherwise the softmax calculation doesn't work.
  const xt::xarray<float> exp_data = xt::exp(modified_input_data);
  const xt::xarray<float> sum_exp_data = xt::sum(exp_data, {-1}, xt::keep_dims);
  const xt::xarray<float> output_data = exp_data / sum_exp_data;

  return output_data;
}

}  // namespace tensorward::util

#pragma once

#include <cassert>
#include <numeric>

#include <xtensor/xarray.hpp>

namespace tensorward::util {

const xt::xarray<float> XtensorSumTo(const xt::xarray<float>& input_data,
                                     const xt::xarray<float>::shape_type& output_shape) {
  // If the input shape is the same as the output shape, then no need to sum.
  if (input_data.shape() == output_shape) {
    return input_data;
  }

  const std::size_t input_dimension = input_data.dimension();
  const std::size_t output_dimension = output_shape.size();
  assert((static_cast<void>("The output dimension must be smaller than or equal to the input dimension."),
         output_dimension <= input_dimension));
  const std::size_t squeeze_axes_size = input_dimension - output_dimension;
  
  // squeeze_axes = {0, 1, 2, ..., squeeze_axes_size - 1}
  xt::xarray<float>::shape_type squeeze_axes(squeeze_axes_size);
  std::iota(squeeze_axes.begin(), squeeze_axes.end(), 0);

  // non_squeeze_axes = {squeeze_axes_size, squeeze_axes_size + 2} for example if output_shape = {1, x, 1}
  xt::xarray<float>::shape_type non_squeeze_axes;
  for (std::size_t i = 0; i <output_shape.size(); ++i) {
    if (output_shape[i] == 1) {
      non_squeeze_axes.push_back(i + squeeze_axes_size);
    }
  }

  // sum_axes = Concat(squeeze_axes, non_squeeze_axes)
  xt::xarray<float>::shape_type sum_axes;
  sum_axes.insert(sum_axes.end(), squeeze_axes.begin(), squeeze_axes.end());
  sum_axes.insert(sum_axes.end(), non_squeeze_axes.begin(), non_squeeze_axes.end());

  xt::xarray<float> output_data = xt::sum(input_data, sum_axes, xt::keep_dims);
  if (1 <= squeeze_axes_size) {
    output_data = xt::squeeze(output_data, squeeze_axes);
  }

  return output_data;
}

}  // namespace tensorward::util

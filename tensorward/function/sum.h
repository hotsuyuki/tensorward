#pragma once

#include <list>
#include <memory>
#include <optional>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class Sum : public core::Function {
 public:
  Sum(const std::optional<xt::xarray<float>::shape_type>& axes_opt = std::nullopt, const bool does_keep_dims = false)
      : core::Function({.num_inputs = 1, .num_outputs = 1}), axes_opt_(axes_opt), does_keep_dims_(does_keep_dims) {}

  ~Sum() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    // NOTE: Ternary operator like `y = does_keep_dims_ ? xt::sum(x, xt::keep_dims) : xt::sum(x);` doesn't work,
    // NOTE: so we use if-else statement instead. 
    xt::xarray<float> y;
    if (axes_opt_.has_value()) {
      if (does_keep_dims_) {
        y = xt::sum(xs[0], axes_opt_.value(), xt::keep_dims);
      } else {
        y = xt::sum(xs[0], axes_opt_.value());
      }
    } else {
      if (does_keep_dims_) {
        y = xt::sum(xs[0], xt::keep_dims);
      } else {
        y = xt::sum(xs[0]);
      }
    }

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    // Create a non-const copy of dL_dy in order to reshape maybe later.
    xt::xarray<float> dL_dy = dL_dys[0];

    // Restores (increases) the dimensions that were reduced (lost) in the forward calculation (summation) by reshaping,
    // if the axes was specified but the keep_dims flag was false in the function argument.
    //
    // e.g. Suppose we have an input array whose shape is {4, 5, 6} with 3 dimensions, and we sum it along with
    // the last axis (-1) and the first axis (0) but without the keep_dims flag (false).
    // Then, the output would be an array whose shape is {5} with 1 dimension.
    //
    //   input_shape: {4, 5, 6} ---> Sum(axes_opt={-1, 0}, does_keep_dims=false) ---> output_shape: {5}
    //
    // This means 2 dimensions were reduced during the forward calculation. The problem is we can't broadcast the shape
    // from {5} to {4, 5, 6} automatically. In order to broadcast to {4, 5, 6}, we need to "convert" {5} to {1, 5, 1}
    // beforehand. This "conversion" is actually restoring the dimensions that were reduced in the forward calculation.
    // Now we can broadcast the shape from {1, 5, 1} to {4, 5, 6} automatically. Problem solved.
    //
    // TODO: Maybe port this code block to tensorward/util as a function so that it can be unit tested ?
    //
    const xt::xarray<float>::shape_type& x_shape = input_tensor_ptrs_[0]->data().shape();
    const std::size_t x_dimension = x_shape.size();
    if (1 <= x_dimension && axes_opt_.has_value() && !does_keep_dims_) {
      // Convert the axes from `xt::xarray<float>::shape_type` to `std::list` in order to be sorted in ascending order.
      // e.g. {-1, 0} ---> {0, 2} ... suppose that the input dimension is 3.
      std::list<int> ascending_axes;
      for (const auto& axis : axes_opt_.value()) {
        ascending_axes.push_back((0 <= axis) ? axis : axis + x_dimension);
      }
      ascending_axes.sort();
      
      // Insert "1" into the shape of dL_dy at the position of the ascending-sorted axes.
      // e.g. {5} ---> {1, 5, 1} ... suppose that the ascending-sorted axes is {0, 2}.
      xt::xarray<float>::shape_type shape = dL_dy.shape();
      for (const auto& ascending_axis : ascending_axes) {
        const auto& insert_position_itr = shape.begin() + ascending_axis;
        shape.insert(insert_position_itr, 1);
      }

      // Restores the dimensions that were reduced in the forward calculation by reshaping.
      dL_dy.reshape(shape);
    }

    const xt::xarray<float> dL_dx = xt::broadcast(dL_dy, x_shape);

    return {dL_dx};
  }

  const std::optional<xt::xarray<float>::shape_type>& axes_opt() const { return axes_opt_; }

  const bool does_keep_dims() const { return does_keep_dims_; }

 private:
  std::optional<xt::xarray<float>::shape_type> axes_opt_;

  bool does_keep_dims_;
};

const core::TensorSharedPtr sum(const core::TensorSharedPtr input_tensor_ptr,
                                const std::optional<xt::xarray<float>::shape_type>& axes_opt = std::nullopt,
                                const bool does_keep_dims = false) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr transpose_function_ptr = std::make_shared<Sum>(axes_opt, does_keep_dims);
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = transpose_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

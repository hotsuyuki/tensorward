#pragma once

#include <memory>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function_fwd.h"
#include "tensorward/core/tensor_fwd.h"

namespace tensorward::core {

class Function : public std::enable_shared_from_this<Function> {
 public:
  Function() {}

  virtual ~Function() {}

  // Performs the forward calculation and the computational graph growth.
  const TensorSharedPtr Call(const TensorSharedPtr input_tensor_ptr);

  // Performs the forward calculation of this function.
  // NOTE: Use this fuction with initialization, instead of assignment, in order to avoid copying the returned value.
  //   * OK: `const xt::xarray<float> y = Forward(x);` ... No copy happens.
  //   * NG: `xt::xarray<float> y;  y = Forward(x);` ... Copy happens.
  virtual const xt::xarray<float> Forward(const xt::xarray<float>& x) const = 0;

  // Performs the backward calculation of this function.
  // NOTE: Use this fuction with initialization, instead of assignment, in order to avoid copying the returned value.
  //   * OK: `const xt::xarray<float> dL_dx = Backward(dL_dy);` ... No copy happens.
  //   * NG: `xt::xarray<float> dL_dx;  dL_dx = Backward(dL_dy);` ... Copy happens.
  virtual const xt::xarray<float> Backward(const xt::xarray<float>& dL_dy) const = 0;

  const TensorSharedPtr input_tensor_ptr() const { return input_tensor_ptr_; }

  const TensorWeakPtr output_tensor_ptr() const { return output_tensor_ptr_; }

 protected:
  TensorSharedPtr input_tensor_ptr_;

  TensorWeakPtr output_tensor_ptr_;
};

}  // namespace tensorward::core

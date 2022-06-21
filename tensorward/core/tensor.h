#pragma once

#include <cassert>
#include <memory>
#include <optional>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function_fwd.h"
#include "tensorward/core/tensor_fwd.h"

namespace tensorward::core {

class Tensor {
 public:
  explicit Tensor(const xt::xarray<float>& data) : data_(data) {}

  // TODO: Change to a virtual destructor when inheriting.
  ~Tensor() {}

  // Starts the backpropagation from this tensor (the last tensor) until the first tensor of the computational graph.
  void Backpropagation();

  // TODO: Maybe add `void SetGrad(xt::xarray<float>& grad)` ?
  void SetGradOpt(const std::optional<xt::xarray<float>>& grad_opt) { grad_opt_ = grad_opt; }

  void SetParentFunctionPtr(const FunctionSharedPtr parent_function_ptr) { parent_function_ptr_ = parent_function_ptr; }

  const xt::xarray<float>& data() const { return data_; }

  const xt::xarray<float>& grad() const {
    assert((static_cast<void>("`Tensor::grad_opt_` must have value to get the value."), grad_opt_.has_value()));
    return grad_opt_.value();
  }

  const std::optional<xt::xarray<float>>& grad_opt() const { return grad_opt_; }

  const FunctionSharedPtr parent_function_ptr() const { return parent_function_ptr_; }

 private:
  xt::xarray<float> data_;

  std::optional<xt::xarray<float>> grad_opt_;

  FunctionSharedPtr parent_function_ptr_;
};

}  // namespace tensorward::core

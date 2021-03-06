#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "tensorward/core/function_fwd.h"
#include "tensorward/core/tensor_fwd.h"

namespace tensorward::core {

class Tensor {
 public:
  Tensor(const xt::xarray<float>& data, const std::string& name = "") : data_(data), name_(name), generation_(0) {}

  virtual ~Tensor() {}

  // Starts the backpropagation from this tensor (the last tensor) until the first tensor in the computational graph.
  void Backpropagation(const bool does_retain_grad = false);

  // Clears the gradient.
  void ClearGrad() { grad_opt_ = std::nullopt; }

  // TODO: Implement `Reshape(output_shape)` by calling `tensorward::function::reshape(this, output_shape)`.

  // TODO: Implement `Transpose()` by calling `tensorward::function::transpose(this)`.

  // TODO: Implement `Sum()` by calling `tensorward::function::sum(this, axes_opt, does_keep_dims)`.

  // TODO: Maybe implement `shape()`, `dimension()` , `size()`, ... etc. by delegating to `xt::xarray<>` functions?

  void SeData(const xt::xarray<float>& data) { data_ = data; }

  // TODO: Maybe add `void SetGrad(xt::xarray<float>& grad)` ?
  void SetGradOpt(const std::optional<xt::xarray<float>>& grad_opt) { grad_opt_ = grad_opt; }

  void SetName(const std::string& name) { name_ = name; }

  void SetParentFunctionPtr(const FunctionSharedPtr parent_function_ptr);

  const xt::xarray<float>& data() const { return data_; }

  const xt::xarray<float>& grad() const {
    assert((static_cast<void>("`Tensor::grad_opt_` must have value to get the value."), grad_opt_.has_value()));
    return grad_opt_.value();
  }

  const std::optional<xt::xarray<float>>& grad_opt() const { return grad_opt_; }

  const std::string& name() const { return name_; }

  const FunctionSharedPtr parent_function_ptr() const { return parent_function_ptr_; }

  const int generation() const { return generation_; }

 protected:
  xt::xarray<float> data_;

  std::optional<xt::xarray<float>> grad_opt_;

  std::string name_;

  FunctionSharedPtr parent_function_ptr_;

  int generation_;
};

const TensorSharedPtr AsTensorSharedPtr(const xt::xarray<float>& data, const std::string& name = "");

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  !tensor.name().empty() ? (os << std::endl << "'" << tensor.name() << "'") : (os << std::endl << "(No name)");
  os << std::endl << "data:" << std::endl << tensor.data();
  if (tensor.grad_opt().has_value()) {
    os << std::endl << "grad:" << std::endl << tensor.grad();
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorSharedPtr tensor_ptr) {
  tensor_ptr ? (os << *tensor_ptr) : (os << std::endl << "(Null)");
  return os;
}

}  // namespace tensorward::core

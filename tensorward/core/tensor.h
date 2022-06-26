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
  Tensor(const xt::xarray<float>& data, const std::string& name = "")
      : data_(data.dimension() == 0 ? xt::xarray<float>({data(0)}) : data),  // Converts to 1-D if it's 0-D scalar.
        name_(name),
        generation_(0) {}

  Tensor(const float data, const std::string& name = "") : Tensor(xt::xarray<float>(data), name) {}

  // TODO: Change to a virtual destructor when inheriting.
  ~Tensor() {}

  // Starts the backpropagation from this tensor (the last tensor) until the first tensor in the computational graph.
  void Backpropagation(const bool does_retain_grad = false);

  // Clears the gradient.
  void ClearGrad() { grad_opt_ = std::nullopt; }

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

 // TODO: Change to `protected:` when inheriting.
 private:
  xt::xarray<float> data_;

  std::optional<xt::xarray<float>> grad_opt_;

  std::string name_;

  FunctionSharedPtr parent_function_ptr_;

  int generation_;
};

const TensorSharedPtr AsTensorSharedPtr(const xt::xarray<float>& data, const std::string& name = "");

const TensorSharedPtr AsTensorSharedPtr(const float data, const std::string& name = "");

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  !tensor.name().empty() ? (os << std::endl << "Tensor '" << tensor.name() << "'") : (os << std::endl << "Tensor");
  os << std::endl << "data:" << std::endl << tensor.data();
  if (tensor.grad_opt().has_value()) {
    os << std::endl << "grad:" << std::endl << tensor.grad();
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const TensorSharedPtr tensor_ptr) {
  tensor_ptr ? (os << *tensor_ptr) : (os << std::endl << "Tensor (Null)");
  return os;
}

}  // namespace tensorward::core

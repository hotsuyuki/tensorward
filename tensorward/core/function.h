#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function_fwd.h"
#include "tensorward/core/tensor_fwd.h"

namespace tensorward::core {

class Function : public std::enable_shared_from_this<Function> {
 public:
  struct NamedArg {
    std::size_t num_inputs;
    std::size_t num_outputs;
  };

  explicit Function(const NamedArg& arg) : num_inputs_(arg.num_inputs), num_outputs_(arg.num_outputs), generation_(0) {}

  virtual ~Function() {}

  // Performs the forward calculation and the computational graph growth.
  const std::vector<TensorSharedPtr> Call(const std::vector<TensorSharedPtr>& input_tensor_ptrs);

  // Performs the forward calculation of this function.
  // NOTE: Use this fuction with initialization, instead of assignment, in order to avoid copying the returned value.
  //   * OK: `const std::vector<xt::xarray<float>> ys = Forward(xs);` ... No copy happens.
  //   * NG: `std::vector<xt::xarray<float>> ys;  ys = Forward(xs);` ... Copy happens.
  // TODO: Maybe change the input argument type to `const std::vector<std::reference_wrapper<xt::xarray<float>>>&` ?
  virtual const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) = 0;

  // Performs the backward calculation of this function.
  // NOTE: Use this fuction with initialization, instead of assignment, in order to avoid copying the returned value.
  //   * OK: `const std::vector<xt::xarray<float>> dL_dxs = Backward(dL_dys);` ... No copy happens.
  //   * NG: `std::vector<xt::xarray<float>> dL_dxs;  dL_dxs = Backward(dL_dys);` ... Copy happens.
  // TODO: Maybe change the input argument type to `const std::vector<std::reference_wrapper<xt::xarray<float>>>&` ?
  virtual const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) = 0;

  const std::size_t num_inputs() const { return num_inputs_; }

  const std::size_t num_outputs() const { return num_outputs_; }

  const std::vector<TensorSharedPtr>& input_tensor_ptrs() const { return input_tensor_ptrs_; }

  const std::vector<TensorWeakPtr>& output_tensor_ptrs() const { return output_tensor_ptrs_; }

  const int generation() const { return generation_; }

 protected:
  std::size_t num_inputs_;

  std::size_t num_outputs_;

  std::vector<TensorSharedPtr> input_tensor_ptrs_;

  std::vector<TensorWeakPtr> output_tensor_ptrs_;

  int generation_;
};

}  // namespace tensorward::core

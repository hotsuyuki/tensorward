#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/layer.h"
#include "tensorward/core/parameter.h"
#include "tensorward/core/tensor.h"
#include "tensorward/function/linear.h"
#include "tensorward/function/matmul.h"

namespace tensorward::layer {

class Linear : public core::Layer {
 public:
  Linear(const std::size_t out_size, const bool does_use_bias = true)
      : out_size_(out_size), does_use_bias_(does_use_bias) {}

  ~Linear() {}

  const std::vector<core::TensorSharedPtr> Forward(
      const std::vector<core::TensorSharedPtr>& input_tensor_ptrs) override {
    const core::TensorSharedPtr x_ptr = input_tensor_ptrs[0];

    if (param_map_.count(W_name_) == 0) {
      // Initializes the weight "W".
      const std::size_t in_size = x_ptr->data().shape(1);
      const float scale = std::sqrt(1.0 / in_size);  // Xavier initialization.
      const core::ParameterSharedPtr W_ptr =
          core::AsParameterSharedPtr(scale * xt::random::randn<float>({in_size, out_size_}), W_name_);
      param_map_[W_name_] = W_ptr;
    }

    if (param_map_.count(b_name_) == 0 && does_use_bias_) {
      // Initializes the bias "b".
      const core::ParameterSharedPtr b_ptr = core::AsParameterSharedPtr(xt::zeros<float>({out_size_}), b_name_);
      param_map_[b_name_] = b_ptr;
    }

    const core::TensorSharedPtr output_tensor_ptr =
        does_use_bias_ ? function::linear(x_ptr, param_map_.at(W_name_), param_map_.at(b_name_))
                       : function::matmul(x_ptr, param_map_.at(W_name_));

    return {output_tensor_ptr};
  }

  const std::size_t out_size() const { return out_size_; }

  const bool does_use_bias() const { return does_use_bias_; }

  const std::string W_name() const { return W_name_; }

  const std::string b_name() const { return b_name_; }

 private:
  std::size_t out_size_;

  bool does_use_bias_;

  const std::string W_name_ = "W";

  const std::string b_name_ = "b";
};

}  // namespace tensorward::layer

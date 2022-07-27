#pragma once

#include <unordered_map>

#include <xtensor/xarray.hpp>

#include "tensorward/core/optimizer.h"
#include "tensorward/core/parameter.h"

namespace tensorward::optimizer {

class MomentumStochasticGradientDescent : public core::Optimizer {
 public:
  MomentumStochasticGradientDescent(const float learning_rate, const float momentum)
      : learning_rate_(learning_rate), momentum_(momentum) {}

  ~MomentumStochasticGradientDescent() {}

  void UpdateSingleParameter(const core::ParameterSharedPtr param_ptr) override {
    // Initializes the velocity of the given parameter as zero if it doesn't exit yet.
    if (velocity_map_.count(param_ptr) == 0) {
      velocity_map_[param_ptr] = xt::zeros_like(param_ptr->data());
    }

    // v <--- m * v - lr * dL_dp
    // p <--- p + v
    velocity_map_[param_ptr] = momentum_ * velocity_map_[param_ptr] - learning_rate_ * param_ptr->grad();
    param_ptr->SeData(param_ptr->data() + velocity_map_[param_ptr]);
  }

  const float learning_rate() const { return learning_rate_; }

  const float momentum() const { return momentum_; }

  const std::unordered_map<core::ParameterSharedPtr, xt::xarray<float>>& velocity_map() const { return velocity_map_; }

 private:
  float learning_rate_;

  float momentum_;

  std::unordered_map<core::ParameterSharedPtr, xt::xarray<float>> velocity_map_;
};

}  // namespace tensorward::optimizer

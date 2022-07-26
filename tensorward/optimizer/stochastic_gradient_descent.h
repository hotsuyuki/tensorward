#pragma once

#include "tensorward/core/optimizer.h"
#include "tensorward/core/parameter.h"

namespace tensorward::optimizer {

class StochasticGradientDescent : public core::Optimizer {
 public:
  StochasticGradientDescent(const float learning_rate) : learning_rate_(learning_rate) {}

  ~StochasticGradientDescent() {}

  void UpdateSingleParameter(const core::ParameterSharedPtr param_ptr) const override {
    param_ptr->SeData(param_ptr->data() - learning_rate_ * param_ptr->grad());
  }

  const float learning_rate() const { return learning_rate_; }

 private:
  float learning_rate_;
};

}  // namespace tensorward::optimizer

#include "tensorward/optimizer/momentum_stochastic_gradient_descent.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::optimizer {

namespace {

constexpr int kInSize = 2;
constexpr int kOutSize = 3;
constexpr float kLearningRate = 0.01;
constexpr float kMomentum = 0.9;
constexpr int kIteration = 10;

}  // namespace

class MomentumStochasticGradientDescentTest : public ::testing::Test {
 protected:
  MomentumStochasticGradientDescentTest()
      : parameter_ptr_(core::AsParameterSharedPtr(xt::random::rand<float>({kInSize, kOutSize}))),
        momentum_stochastic_gradient_descent_optimizer_(kLearningRate, kMomentum) {}

  const core::ParameterSharedPtr parameter_ptr_;
  MomentumStochasticGradientDescent momentum_stochastic_gradient_descent_optimizer_;
};

TEST_F(MomentumStochasticGradientDescentTest, UpdateSingleParameterTest) {
  // Sets a pseudo gradient.
  const xt::xarray<float> dL_dp = xt::ones_like(parameter_ptr_->data());
  parameter_ptr_->SetGradOpt(dL_dp);

  xt::xarray<float> expected_parameter_data = parameter_ptr_->data();
  xt::xarray<float> velocity = xt::zeros_like(parameter_ptr_->data());
  for (std::size_t i = 0; i < kIteration; ++i) {
    // Sets an expected parameter data according to the Momentum Stochastic Gradient Decent equation.
    // v <--- m * v - lr * dL_dp
    // p <--- p + v
    velocity = kMomentum * velocity - kLearningRate * dL_dp;
    expected_parameter_data = expected_parameter_data + velocity;
  }
 
  // We need to run multiple times to have a positive "velocity" value during the parameter update. In other words,
  // the "velocity" is zero if we just run a single time (and it's equivalent to normal Stochastic Gradient Decent).
  for (std::size_t i = 0; i < kIteration; ++i) {
    momentum_stochastic_gradient_descent_optimizer_.UpdateSingleParameter(parameter_ptr_);
  }

  // Prepares actual parameter datas.
  const xt::xarray<float> actual_parameter_data = parameter_ptr_->data();

  // Checks that the updated parameters are correct.
  EXPECT_EQ(actual_parameter_data, expected_parameter_data);
}

}  // namespace tensorward::optimizer

#include "tensorward/optimizer/stochastic_gradient_descent.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

namespace tensorward::optimizer {

namespace {

constexpr int kInSize = 2;
constexpr int kOutSize = 3;
constexpr float kLearningRate = 0.01;

}  // namespace

class StochasticGradientDescentTest : public ::testing::Test {
 protected:
  StochasticGradientDescentTest()
      : parameter_ptr_(core::AsParameterSharedPtr(xt::random::rand<float>({kInSize, kOutSize}))),
        stochastic_gradient_descent_optimizer_(kLearningRate) {}

  const core::ParameterSharedPtr parameter_ptr_;
  StochasticGradientDescent stochastic_gradient_descent_optimizer_;
};

TEST_F(StochasticGradientDescentTest, UpdateSingleParameterTest) {
  // Sets a pseudo gradient.
  const xt::xarray<float> dL_dp = xt::ones_like(parameter_ptr_->data());
  parameter_ptr_->SetGradOpt(dL_dp);

  // Sets an expected parameter data according to the Stochastic Gradient Decent equation.
  // p <--- p - lr * dL_dp
  const xt::xarray<float> expected_parameter_data = parameter_ptr_->data() - kLearningRate * dL_dp;

  stochastic_gradient_descent_optimizer_.UpdateSingleParameter(parameter_ptr_);

  // Prepares actual parameter datas.
  const xt::xarray<float> actual_parameter_data = parameter_ptr_->data();

  // Checks that the updated parameters are correct.
  EXPECT_EQ(actual_parameter_data, expected_parameter_data);
}

}  // namespace tensorward::optimizer

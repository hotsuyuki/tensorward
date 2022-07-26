#include "tensorward/core/optimizer.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/optimizer/stochastic_gradient_descent.h"

namespace tensorward::core {

namespace {

constexpr int kInSize = 2;
constexpr int kOutSize = 3;
constexpr float kLearningRate = 0.01;

}  // namespace

class OptimizerTest : public ::testing::Test {
 protected:
  OptimizerTest()
      : W_ptr_(AsParameterSharedPtr(xt::random::rand<float>({kInSize, kOutSize}))),
        b_ptr_(AsParameterSharedPtr(xt::random::rand<float>({kOutSize}))),
        parameter_ptrs_({W_ptr_, b_ptr_}),
        stochastic_gradient_descent_optimizer_(kLearningRate) {}

  const ParameterSharedPtr W_ptr_;
  const ParameterSharedPtr b_ptr_;
  const std::vector<ParameterSharedPtr> parameter_ptrs_;
  const optimizer::StochasticGradientDescent stochastic_gradient_descent_optimizer_;
};

TEST_F(OptimizerTest, UpdateTest) {
  // Prepares expected parameter datas.
  std::vector<xt::xarray<float>> expected_parameter_datas;
  expected_parameter_datas.reserve(parameter_ptrs_.size());

  for (const auto& parameter_ptr : parameter_ptrs_) {
    // Sets a pseudo gradient.
    const xt::xarray<float> dL_dp = xt::ones_like(parameter_ptr->data());
    parameter_ptr->SetGradOpt(dL_dp);

    // Sets an expected parameter data according to the Stochastic Gradient Decent equation.
    // p <--- p - lr * dL_dp
    const xt::xarray<float> expected_parameter_data = parameter_ptr->data() - kLearningRate * dL_dp;
    expected_parameter_datas.push_back(expected_parameter_data);
  }

  stochastic_gradient_descent_optimizer_.Update(parameter_ptrs_);

  // Prepares actual parameter datas.
  std::vector<xt::xarray<float>> actual_parameter_datas;
  actual_parameter_datas.reserve(parameter_ptrs_.size());
  for (const auto& parameter_ptr : parameter_ptrs_) {
    actual_parameter_datas.push_back(parameter_ptr->data());
  }

  // Checks that the updated parameters are correct.
  ASSERT_EQ(actual_parameter_datas.size(), expected_parameter_datas.size());
  for (std::size_t i = 0; i < expected_parameter_datas.size(); ++i) {
    EXPECT_EQ(actual_parameter_datas[i], expected_parameter_datas[i]);
  }
}

}  // namespace tensorward::core

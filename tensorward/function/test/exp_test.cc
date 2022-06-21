#include "tensorward/function/exp.h"

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/util/numerical_gradient.h"

namespace tensorward::function {

namespace {

constexpr unsigned int kHight = 2;
constexpr unsigned int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;

}  // namespace

class ExpTest : public ::testing::Test {
 protected:
  ExpTest()
      : input_tensor_ptr_(std::make_shared<core::Tensor>(xt::random::rand<float>({kHight, kWidth}))),
        output_tensor_ptr_(exp(input_tensor_ptr_)) {}

  const core::TensorSharedPtr input_tensor_ptr_;
  const core::TensorSharedPtr output_tensor_ptr_;
};

TEST_F(ExpTest, AnalyticalForwardTest) {
  const xt::xarray<float>& actual_output_data = output_tensor_ptr_->data();

  xt::xarray<float> expected_output_data(input_tensor_ptr_->data());
  std::for_each(expected_output_data.begin(), expected_output_data.end(), [](float& elem) { elem = std::exp(elem); });

  // Checks that the forward calculation is correct (analytically).
  EXPECT_EQ(actual_output_data, expected_output_data);
}

TEST_F(ExpTest, AnalyticalBackwardTest) {
  const core::FunctionSharedPtr exp_function_ptr = output_tensor_ptr_->parent_function_ptr();
  const xt::xarray<float> actual_input_grad = exp_function_ptr->Backward(xt::ones_like(output_tensor_ptr_->data()));

  xt::xarray<float> expected_input_grad(input_tensor_ptr_->data());
  std::for_each(expected_input_grad.begin(), expected_input_grad.end(), [](float& elem) { elem = std::exp(elem); });

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grad, expected_input_grad);
}

TEST_F(ExpTest, NumericalBackwardTest) {
  const core::FunctionSharedPtr exp_function_ptr = output_tensor_ptr_->parent_function_ptr();
  const xt::xarray<float> actual_input_grad = exp_function_ptr->Backward(xt::ones_like(output_tensor_ptr_->data()));

  const core::FunctionSharedPtr new_exp_function_ptr = std::make_shared<Exp>();
  const xt::xarray<float>& input_data = input_tensor_ptr_->data();
  const xt::xarray<float> expected_input_grad = util::NumericalGradient(new_exp_function_ptr, input_data, kEpsilon);

  // Checks that the backward calculation is correct (numerically).
  ASSERT_EQ(actual_input_grad.shape(), expected_input_grad.shape());
  ASSERT_EQ(actual_input_grad.shape(0), kHight);
  ASSERT_EQ(actual_input_grad.shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_input_grad(i, j), expected_input_grad(i, j), kEpsilon);
    }
  }
}

}  // namespace tensorward::function

#include "tensorward/util/numerical_gradient.h"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/function/exp.h"
#include "tensorward/function/square.h"

namespace tensorward::util {

namespace {

constexpr unsigned int kHight = 2;
constexpr unsigned int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;

}  // namespace

class NumericalGradientTest : public ::testing::Test {
 protected:
  NumericalGradientTest() : input_data_(xt::random::rand<float>({kHight, kWidth})) {}

  const xt::xarray<float> input_data_;
};

TEST_F(NumericalGradientTest, ExpTest) {
  const core::FunctionSharedPtr exp_function_ptr = std::make_shared<function::Exp>();
  const xt::xarray<float> actual_grad = NumericalGradient(exp_function_ptr, input_data_, kEpsilon);

  xt::xarray<float> expected_grad(input_data_);
  std::for_each(expected_grad.begin(), expected_grad.end(), [](float& elem) { elem = std::exp(elem); });

  // Checks that the numerical gradient is close to the analytical gradient.
  ASSERT_EQ(actual_grad.shape(), expected_grad.shape());
  ASSERT_EQ(actual_grad.shape(0), kHight);
  ASSERT_EQ(actual_grad.shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_grad(i, j), expected_grad(i, j), kEpsilon);
    }
  }
}

TEST_F(NumericalGradientTest, SquareTest) {
  const core::FunctionSharedPtr square_function_ptr = std::make_shared<function::Square>();
  const xt::xarray<float> actual_grad = NumericalGradient(square_function_ptr, input_data_, kEpsilon);

  xt::xarray<float> expected_grad(input_data_);
  std::for_each(expected_grad.begin(), expected_grad.end(), [](float& elem) { elem = 2.0f * elem; });

  // Checks that the numerical gradient is close to the analytical gradient.
  ASSERT_EQ(actual_grad.shape(), expected_grad.shape());
  ASSERT_EQ(actual_grad.shape(0), kHight);
  ASSERT_EQ(actual_grad.shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_grad(i, j), expected_grad(i, j), kEpsilon);
    }
  }
}
  
}  // namespace tensorward::util

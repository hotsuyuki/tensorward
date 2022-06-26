#include "tensorward/util/numerical_gradient.h"

#include <cmath>
#include <memory>

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/function/exp.h"
#include "tensorward/function/pow.h"
#include "tensorward/function/square.h"

namespace tensorward::util {

namespace {

constexpr int kHight = 2;
constexpr int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;
constexpr int kExponentLower = -5;
constexpr int kExponentUpper = 5;

}  // namespace

class NumericalGradientTest : public ::testing::Test {
 protected:
  NumericalGradientTest()
      : input_data0_(xt::random::rand<float>({kHight, kWidth})),
        input_data1_(xt::random::rand<float>({kHight, kWidth})) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data1_;
};

TEST_F(NumericalGradientTest, ExpTest) {
  const core::FunctionSharedPtr exp_function_ptr = std::make_shared<function::Exp>();
  const std::vector<xt::xarray<float>> input_datas({input_data0_});
  const std::vector<xt::xarray<float>> actual_grads = NumericalGradient(exp_function_ptr, input_datas, kEpsilon);
  ASSERT_EQ(actual_grads.size(), 1);

  xt::xarray<float> expected_grad(input_data0_);
  std::for_each(expected_grad.begin(), expected_grad.end(), [](float& elem) { elem = std::exp(elem); });

  // Checks that the numerical gradient is close to the analytical gradient.
  ASSERT_EQ(actual_grads[0].shape(), expected_grad.shape());
  ASSERT_EQ(actual_grads[0].shape(0), kHight);
  ASSERT_EQ(actual_grads[0].shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_grads[0](i, j), expected_grad(i, j), kEpsilon);
    }
  }
}

TEST_F(NumericalGradientTest, PowTest) {
  for (int exponent = kExponentLower; exponent < kExponentUpper; ++exponent) {
    const core::FunctionSharedPtr pow_function_ptr = std::make_shared<function::Pow>(exponent);
    const std::vector<xt::xarray<float>> input_datas({input_data0_});
    const std::vector<xt::xarray<float>> actual_grads = NumericalGradient(pow_function_ptr, input_datas, kEpsilon);
    ASSERT_EQ(actual_grads.size(), 1);

    xt::xarray<float> expected_grad(input_data0_);
    std::for_each(expected_grad.begin(), expected_grad.end(),
                  [exponent](float& elem) { elem = exponent * std::pow(elem, exponent - 1); });

    // Checks that the numerical gradient is close to the analytical gradient.
    ASSERT_EQ(actual_grads[0].shape(), expected_grad.shape());
    ASSERT_EQ(actual_grads[0].shape(0), kHight);
    ASSERT_EQ(actual_grads[0].shape(1), kWidth);
    for (std::size_t i = 0; i < kHight; ++i) {
      for (std::size_t j = 0; j < kWidth; ++j) {
        // Sets the tolerance as (100 * kEpsilon)% of the expected value.
        const float tolerance = std::abs(expected_grad(i, j) * kEpsilon);
        EXPECT_NEAR(actual_grads[0](i, j), expected_grad(i, j), tolerance);
      }
    }
  }
}

TEST_F(NumericalGradientTest, SquareTest) {
  const core::FunctionSharedPtr square_function_ptr = std::make_shared<function::Square>();
  const std::vector<xt::xarray<float>> input_datas({input_data0_});
  const std::vector<xt::xarray<float>> actual_grads = NumericalGradient(square_function_ptr, input_datas, kEpsilon);
  ASSERT_EQ(actual_grads.size(), 1);

  xt::xarray<float> expected_grad(input_data0_);
  std::for_each(expected_grad.begin(), expected_grad.end(), [](float& elem) { elem = 2.0 * elem; });

  // Checks that the numerical gradient is close to the analytical gradient.
  ASSERT_EQ(actual_grads[0].shape(), expected_grad.shape());
  ASSERT_EQ(actual_grads[0].shape(0), kHight);
  ASSERT_EQ(actual_grads[0].shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_grads[0](i, j), expected_grad(i, j), kEpsilon);
    }
  }
}

}  // namespace tensorward::util

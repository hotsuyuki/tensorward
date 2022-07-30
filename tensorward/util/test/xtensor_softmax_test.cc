#include "tensorward/util/xtensor_softmax.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::util {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;

}  // namespace

class XtensorSoftmaxTest : public ::testing::Test {
 protected:
  XtensorSoftmaxTest()
      : input_data_(xt::random::rand<float>({kHeight, kWidth})) {}

  const xt::xarray<float> input_data_;
};

TEST_F(XtensorSoftmaxTest, SoftmaxTest) {
  const xt::xarray<float> actual_output_data = XtensorSoftmax(input_data_);

  const xt::xarray<float> expected_exp_data = xt::exp(input_data_);
  const xt::xarray<float> expected_sum_exp_data = xt::sum(expected_exp_data, {-1}, xt::keep_dims);
  const xt::xarray<float> expected_output_data = expected_exp_data / expected_sum_exp_data;

  // Checks that the calculation is correct.
  ASSERT_EQ(actual_output_data.shape(), expected_output_data.shape());
  ASSERT_EQ(actual_output_data.shape(0), kHeight);
  ASSERT_EQ(actual_output_data.shape(1), kWidth);
  for (std::size_t i = 0; i < kHeight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_output_data(i, j), expected_output_data(i, j), kEpsilon);
    }
  }
}

}  // namespace tensorward::util

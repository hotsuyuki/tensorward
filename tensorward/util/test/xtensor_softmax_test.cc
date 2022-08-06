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
      : input_data_(xt::random::rand<float>({kHeight, kWidth})),
        input_data_exp_(xt::exp(input_data_)),                               // NOTE: Need to construct for softmax.
        input_data_sum_exp_(xt::sum(input_data_exp_, {-1}, xt::keep_dims)),  // NOTE: Need to construct for softmax.
        expected_output_data_(input_data_exp_ / input_data_sum_exp_) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float> input_data_exp_;
  const xt::xarray<float> input_data_sum_exp_;
  const xt::xarray<float> expected_output_data_;
};

TEST_F(XtensorSoftmaxTest, SoftmaxTest) {
  const xt::xarray<float> actual_output_data = XtensorSoftmax(input_data_);

  // Checks that the calculation is correct.
  ASSERT_EQ(actual_output_data.shape(), expected_output_data_.shape());
  ASSERT_EQ(actual_output_data.shape(0), kHeight);
  ASSERT_EQ(actual_output_data.shape(1), kWidth);
  for (std::size_t i = 0; i < kHeight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_output_data(i, j), expected_output_data_(i, j), kEpsilon);
    }
  }
}

}  // namespace tensorward::util

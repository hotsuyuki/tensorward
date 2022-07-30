#include "tensorward/util/xtensor_cross_entropy_error.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::util {

namespace {

constexpr int kDataSize = 2;
constexpr int kOutSize = 3;

}  // namespace

class XtensorCrossEntropyErrorTest : public ::testing::Test {
 protected:
  XtensorCrossEntropyErrorTest()
      : input_data0_(xt::random::rand<float>({kDataSize, kOutSize})),                // p
        input_data1_onehot_(xt::xarray<float>({{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}})),  // t (onehot)
        input_data1_non_onehot_(xt::xarray<float>({0.0, 2.0})),                      // t (non-onehot)
        expected_output_data_(-xt::sum(input_data1_onehot_ * xt::log(input_data0_)) / kDataSize) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data1_onehot_;
  const xt::xarray<float> input_data1_non_onehot_;
  const xt::xarray<float> expected_output_data_;
};

TEST_F(XtensorCrossEntropyErrorTest, CrossEntropyErrorWithOnehotLabelTest) {
  const xt::xarray<float> actual_output_data = XtensorCrossEntropyError(input_data0_, input_data1_onehot_);

  // Checks that the calculation is correct.
  EXPECT_EQ(actual_output_data, expected_output_data_);
}

TEST_F(XtensorCrossEntropyErrorTest, CrossEntropyErrorWithNonOnehotLabelTest) {
  const xt::xarray<float> actual_output_data = XtensorCrossEntropyError(input_data0_, input_data1_non_onehot_);

  // Checks that the calculation is correct.
  EXPECT_EQ(actual_output_data, expected_output_data_);
}

}  // namespace tensorward::util

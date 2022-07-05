#include "tensorward/util/xtensor_sum_to.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::util {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class XtensorSumToTest : public ::testing::Test {
 protected:
  XtensorSumToTest()
      : input_data_(xt::random::rand<float>({kHeight, kWidth})) {}

  const xt::xarray<float> input_data_;
};

TEST_F(XtensorSumToTest, SumAlongWithAxis0Test) {
  const xt::xarray<float>::shape_type output_shape_2D_1_kWidth({1, kWidth});
  const xt::xarray<float> actual_output_data_2D_1_kWidth = XtensorSumTo(input_data_, output_shape_2D_1_kWidth);
  const xt::xarray<float> expected_output_data_2D_1_kWidth = xt::sum(input_data_, {0}, xt::keep_dims);

  // expected = {{a, b, c, ..., w}} with 2D.
  EXPECT_EQ(actual_output_data_2D_1_kWidth.shape(), output_shape_2D_1_kWidth);
  EXPECT_EQ(actual_output_data_2D_1_kWidth, expected_output_data_2D_1_kWidth);

  const xt::xarray<float>::shape_type output_shape_1D_kWidth({kWidth});
  const xt::xarray<float> actual_output_data_1D_kWidth = XtensorSumTo(input_data_, output_shape_1D_kWidth);
  const xt::xarray<float> expected_output_data_1D_kWidth = xt::row(expected_output_data_2D_1_kWidth, 0);

  // expected = {a, b, c, ..., w} with 1D.
  EXPECT_EQ(actual_output_data_1D_kWidth.shape(), output_shape_1D_kWidth);
  EXPECT_EQ(actual_output_data_1D_kWidth, expected_output_data_1D_kWidth);
}

TEST_F(XtensorSumToTest, SumAlongWithAxis1Test) {
  const xt::xarray<float>::shape_type output_shape_2D_kHeight_1({kHeight, 1});
  const xt::xarray<float> actual_output_data_2D_kHeight_1 = XtensorSumTo(input_data_, output_shape_2D_kHeight_1);
  const xt::xarray<float> expected_output_data_2D_kHeight_1 = xt::sum(input_data_, {1}, xt::keep_dims);

  // expected = {{a},
  //             {b},
  //             {c},
  //             ...,
  //             {h}} with 2D.
  EXPECT_EQ(actual_output_data_2D_kHeight_1.shape(), output_shape_2D_kHeight_1);
  EXPECT_EQ(actual_output_data_2D_kHeight_1, expected_output_data_2D_kHeight_1);
}

TEST_F(XtensorSumToTest, SumAlongWithAxis0and1Test) {
  const xt::xarray<float>::shape_type output_shape_2D_1_1({1, 1});
  const xt::xarray<float> actual_output_data_2D_1_1 = XtensorSumTo(input_data_, output_shape_2D_1_1);
  const xt::xarray<float> expected_output_data_2D_1_1 = xt::sum(input_data_, {0, 1}, xt::keep_dims);

  // expected = {{a}} with 2D.
  EXPECT_EQ(actual_output_data_2D_1_1.shape(), output_shape_2D_1_1);
  EXPECT_EQ(actual_output_data_2D_1_1, expected_output_data_2D_1_1);

  const xt::xarray<float>::shape_type output_shape_1D_1({1});
  const xt::xarray<float> actual_output_data_1D_1 = XtensorSumTo(input_data_, output_shape_1D_1);
  const xt::xarray<float> expected_output_data_1D_1 = xt::row(expected_output_data_2D_1_1, 0);

  // expected = {a} with 1D.
  EXPECT_EQ(actual_output_data_1D_1.shape(), output_shape_1D_1);
  EXPECT_EQ(actual_output_data_1D_1, expected_output_data_1D_1);

  const xt::xarray<float>::shape_type output_shape_0D({});
  const xt::xarray<float> actual_output_data_0D = XtensorSumTo(input_data_, output_shape_0D);
  const xt::xarray<float> expected_output_data_0D = expected_output_data_2D_1_1(0, 0);

  // expected = a with 0D.
  EXPECT_EQ(actual_output_data_0D.shape(), output_shape_0D);
  EXPECT_EQ(actual_output_data_0D, expected_output_data_0D);
}

}  // namespace tensorward::util

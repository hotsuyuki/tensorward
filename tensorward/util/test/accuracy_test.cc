#include "tensorward/util/accuracy.h"

#include <gtest/gtest.h>

namespace tensorward::util {

class AccuracyTest : public ::testing::Test {
 protected:
  AccuracyTest()
      // clang-format off
      : score_one_third_correct_(xt::xarray<float>({{4.2, 0.0, -1.0},
                                                    {4.2, 0.0, -1.0},
                                                    {4.2, 0.0, -1.0}})),
        score_two_thirds_correct_(xt::xarray<float>({{4.2, 0.0, -1.0},
                                                     {-1.0, 4.2, 0.0},
                                                     {-1.0, 4.2, 0.0}})),
        score_three_thirds_correct_(xt::xarray<float>({{4.2, 0.0, -1.0},
                                                       {-1.0, 4.2, 0.0},
                                                       {0.0, -1.0, 4.2}})),
        label_onehot_(xt::xarray<float>({{1.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0}})),
        label_non_onehot_(xt::xarray<float>({0.0, 1.0, 2.0})) {}
      // clang-format on

  const xt::xarray<float> score_one_third_correct_;
  const xt::xarray<float> score_two_thirds_correct_;
  const xt::xarray<float> score_three_thirds_correct_;
  const xt::xarray<float> label_onehot_;
  const xt::xarray<float> label_non_onehot_;
};

TEST_F(AccuracyTest, Test) {
  // Tests with both onehot label and non-onehot label.
  for (const auto& label : {label_onehot_, label_non_onehot_}) {
    // Checks that the accuracy calculation is correct.
    EXPECT_EQ(Accuracy(score_one_third_correct_, label), 1.0f / 3.0f);
    EXPECT_EQ(Accuracy(score_two_thirds_correct_, label), 2.0f / 3.0f);
    EXPECT_EQ(Accuracy(score_three_thirds_correct_, label), 3.0f / 3.0f);
  }
}

}  // namespace tensorward::util

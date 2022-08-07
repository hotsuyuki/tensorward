#include "tensorward/core/dataset.h"

#include <cassert>

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "tensorward/dataset/spiral.h"

namespace tensorward::core {

namespace {

constexpr std::size_t kClassSize = 3;
constexpr bool kIsTrainingMode = true;

}  // namespace

class DatasetTest : public ::testing::Test {
 protected:
  DatasetTest()
      : make_it_half_lambda_([](const xt::xarray<float>& input_data) { return input_data / 2.0; }),
        make_it_onehot_lambda_([](const xt::xarray<float>& input_data) {
          const std::size_t class_index = static_cast<std::size_t>(input_data(0));
          assert(class_index < kClassSize);
          xt::xarray<float> output_data = xt::zeros<float>({kClassSize});
          xt::view(output_data, class_index) = 1.0;
          return output_data;
        }),
        data_transform_lambdas_({make_it_half_lambda_}),
        label_transform_lambdas_({make_it_onehot_lambda_}),
        spiral_dataset_(kIsTrainingMode, data_transform_lambdas_, label_transform_lambdas_) {}

  const TransformLambda make_it_half_lambda_;
  const TransformLambda make_it_onehot_lambda_;
  const std::vector<TransformLambda> data_transform_lambdas_;
  const std::vector<TransformLambda> label_transform_lambdas_;
  const dataset::Spiral spiral_dataset_;
};

TEST_F(DatasetTest, SizeTest) {
  // Calls the base class's member function to get the actual dataset size.
  const std::size_t actual_dataset_size = spiral_dataset_.size();

  // Calls the derived class's member function to get the expected dataset size.
  const std::size_t expected_dataset_size = spiral_dataset_.data_size();

  // Checks that the dataset size is correct.
  EXPECT_EQ(actual_dataset_size, expected_dataset_size);
}

TEST_F(DatasetTest, AtTest) {
  for (std::size_t i = 0; i < spiral_dataset_.size(); ++ i) {
    // Prepares the actual i-th data and label, which is expected to be transformed already.
    const auto [actual_ith_data, actual_ith_label] = spiral_dataset_.at(i);

    // Uses the copy construct in order to avoid modifying the original data and label when applying transform lambdas.
    xt::xarray<float> ith_data(xt::view(spiral_dataset_.data(), i));
    xt::xarray<float> ith_label(xt::view(spiral_dataset_.label(), i));

    // Prepares the expected i-th data, and applies the "make-it-half" transform to it.
    xt::xarray<float> expected_ith_data = ith_data;
    expected_ith_data = expected_ith_data / 2.0;

    // Prepares the expected i-th label, and applies the "make-it-onehot" transform to it.
    const std::size_t class_index = static_cast<std::size_t>(ith_label(0));
    assert(class_index < kClassSize);
    xt::xarray<float> expected_ith_label = xt::zeros<float>({kClassSize});
    xt::view(expected_ith_label, class_index) = 1.0;

    // Checks that the obtained i-th data and label are correct.
    EXPECT_EQ(actual_ith_data, expected_ith_data);
    EXPECT_EQ(actual_ith_label, expected_ith_label);
  }
}

}  // namespace tensorward::core

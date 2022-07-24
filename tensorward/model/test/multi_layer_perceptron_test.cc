#include "tensorward/model/multi_layer_perceptron.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/parameter.h"
#include "tensorward/function/sigmoid.h"

namespace tensorward::model {

namespace {

constexpr int kDataSize = 10;
constexpr int kInSize = 2;
constexpr int kHiddenSize = 5;
constexpr int kOutSize = 3;

}  // namespace

class MultiLayerPerceptronTest : public ::testing::Test {
 protected:
  MultiLayerPerceptronTest()
      : input_tensor_ptr_(core::AsTensorSharedPtr(xt::random::rand<float>({kDataSize, kInSize}))) {}

  const core::TensorSharedPtr input_tensor_ptr_;
};

TEST_F(MultiLayerPerceptronTest, PredictTest) {
  // y = linear(sigmoid(linear(x)))
  MultiLayerPerceptron multi_layer_perceptron_model({kHiddenSize, kOutSize}, function::sigmoid_lambda);

  const std::vector<core::TensorSharedPtr> actual_input_tensor_ptrs({input_tensor_ptr_});
  const std::vector<core::TensorSharedPtr> actual_output_tensor_ptrs =
      multi_layer_perceptron_model.Predict(actual_input_tensor_ptrs);
  ASSERT_EQ(actual_output_tensor_ptrs.size(), 1);
  const xt::xarray<float>& actual_output_data = actual_output_tensor_ptrs[0]->data();

  // There should exist 2 layers: "layer0", "layer1".
  ASSERT_EQ(multi_layer_perceptron_model.layer_ptrs().size(), 2);

  // Identifies the obtained parameter from layer0 is either weight "W0" or bias "b0".
  xt::xarray<float> W0_data;
  xt::xarray<float> b0_data;
  const xt::xarray<float>::shape_type W0_shape({kInSize, kHiddenSize});
  const xt::xarray<float>::shape_type b0_shape({kHiddenSize});
  const core::LayerSharedPtr linear_layer0_ptr = multi_layer_perceptron_model.layer_ptrs()[0];
  for (const auto& param0_name_ptr : linear_layer0_ptr->param_map()) {
    const core::ParameterSharedPtr param0_ptr = param0_name_ptr.second;
    ASSERT_TRUE(param0_ptr->data().shape() == W0_shape || param0_ptr->data().shape() == b0_shape);
    if (param0_ptr->data().shape() == W0_shape) {
      W0_data = param0_ptr->data();
    } else if (param0_ptr->data().shape() == b0_shape) {
      b0_data = param0_ptr->data();
    }
  }

  // Identifies the obtained parameter from layer1 is either weight "W1" or bias "b1".
  xt::xarray<float> W1_data;
  xt::xarray<float> b1_data;
  const xt::xarray<float>::shape_type W1_shape({kHiddenSize, kOutSize});
  const xt::xarray<float>::shape_type b1_shape({kOutSize});
  const core::LayerSharedPtr linear_layer1_ptr = multi_layer_perceptron_model.layer_ptrs()[1];
  for (const auto& param1_name_ptr : linear_layer1_ptr->param_map()) {
    const core::ParameterSharedPtr param1_ptr = param1_name_ptr.second;
    ASSERT_TRUE(param1_ptr->data().shape() == W1_shape || param1_ptr->data().shape() == b1_shape);
    if (param1_ptr->data().shape() == W1_shape) {
      W1_data = param1_ptr->data();
    } else if (param1_ptr->data().shape() == b1_shape) {
      b1_data = param1_ptr->data();
    }
  }

  xt::xarray<float> expected_output_data = input_tensor_ptr_->data();
  expected_output_data = xt::linalg::dot(expected_output_data, W0_data) + b0_data;  //  linear: x W + b
  expected_output_data = 1.0 / (1.0 + xt::exp(-expected_output_data));              // sigmoid: 1 / (1 + exp(-x))
  expected_output_data = xt::linalg::dot(expected_output_data, W1_data) + b1_data;  //  linear: x W + b

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_data, expected_output_data);
}

}  // namespace tensorward::model

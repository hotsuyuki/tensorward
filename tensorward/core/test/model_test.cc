#include "tensorward/core/model.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/function/sigmoid.h"
#include "tensorward/model/multi_layer_perceptron.h"

namespace tensorward::core {

namespace {

constexpr int kDataSize = 10;
constexpr int kInSize = 2;
constexpr int kHiddenSize = 5;
constexpr int kOutSize = 3;

}  // namespace

class ModelTest : public ::testing::Test {
 protected:
  ModelTest()
      : input_tensor_ptr_(AsTensorSharedPtr(xt::random::rand<float>({kDataSize, kInSize}))),
        multi_layer_perceptron_model_(model::MultiLayerPerceptron({kHiddenSize, kOutSize}, function::sigmoid_lambda)),
        output_tensor_ptr_(multi_layer_perceptron_model_.Predict({input_tensor_ptr_})[0]) {}

  const TensorSharedPtr input_tensor_ptr_;
  model::MultiLayerPerceptron multi_layer_perceptron_model_;
  const TensorSharedPtr output_tensor_ptr_;
};

TEST_F(ModelTest, ClearGradsTest) {
  output_tensor_ptr_->Backpropagation();

  // Checks that parameter tensors (and input tensors) have their gradient after backpropagation.
  for (const auto& param_ptr : multi_layer_perceptron_model_.GetParamPtrs()) {
    EXPECT_TRUE(param_ptr->grad_opt().has_value());
  }
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());

  multi_layer_perceptron_model_.ClearGrads();

  // Checks that parameter tensors don't have (but input tensors still have) their gradient after gradient clearing.
  for (const auto& param_ptr : multi_layer_perceptron_model_.GetParamPtrs()) {
    EXPECT_FALSE(param_ptr->grad_opt().has_value());
  }
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());
}

TEST_F(ModelTest, GetParamPtrsTest) {
  const std::vector<ParameterSharedPtr> param_ptrs = multi_layer_perceptron_model_.GetParamPtrs();

  // There should exist 4 parameters:
  //   * from layer0 ... weight "W0", bias "b0"
  //   * from layer1 ... weight "W1", bias "b1"
  ASSERT_EQ(param_ptrs.size(), 4);

  const xt::xarray<float>::shape_type expected_W0_shape({kInSize, kHiddenSize});
  const xt::xarray<float>::shape_type expected_b0_shape({kHiddenSize});

  // Checks that the shape of the parameters from layer0 (weight "W0", bias "b0") is correct.
  // NOTE: We can't identify the obtained parameter from layer0 is either weight "W0" or bias "b0" (due to the nature of
  // NOTE: the range loop of `std::unordered_map`), so we check both by Logical OR.
  ASSERT_TRUE(param_ptrs[0]->data().shape() == expected_W0_shape || param_ptrs[0]->data().shape() == expected_b0_shape);
  if (param_ptrs[0]->data().shape() == expected_W0_shape) {
    // If the first obtained parameter is weight "W0", then the second obtained parameter would be bias "b0".
    EXPECT_TRUE(param_ptrs[1]->data().shape() == expected_b0_shape);
  } else if (param_ptrs[0]->data().shape() == expected_b0_shape) {
    // If the first obtained parameter is bias "b0", then the second obtained parameter would be weight "W0".
    EXPECT_TRUE(param_ptrs[1]->data().shape() == expected_W0_shape);
  }

  const xt::xarray<float>::shape_type expected_W1_shape({kHiddenSize, kOutSize});
  const xt::xarray<float>::shape_type expected_b1_shape({kOutSize});

  // Checks that the shape of the parameters from layer1 (weight "W1", bias "b1") is correct.
  // NOTE: We can't identify the obtained parameter from layer1 is either weight "W1" or bias "b1" (due to the nature of
  // NOTE: the range loop of `std::unordered_map`), so we check both by Logical OR.
  ASSERT_TRUE(param_ptrs[2]->data().shape() == expected_W1_shape || param_ptrs[2]->data().shape() == expected_b1_shape);
  if (param_ptrs[2]->data().shape() == expected_W1_shape) {
    // If the first obtained parameter is weight "W1", then the second obtained parameter would be bias "b1".
    EXPECT_TRUE(param_ptrs[3]->data().shape() == expected_b1_shape);
  } else if (param_ptrs[2]->data().shape() == expected_b1_shape) {
    // If the first obtained parameter is bias "b1", then the second obtained parameter would be weight "W1".
    EXPECT_TRUE(param_ptrs[3]->data().shape() == expected_W1_shape);
  }
}

}  // namespace tensorward::core

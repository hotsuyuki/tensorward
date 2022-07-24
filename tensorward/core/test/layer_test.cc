#include "tensorward/core/layer.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/layer/linear.h"

namespace tensorward::core {

namespace {

constexpr int kDataSize = 10;
constexpr int kInSize = 2;
constexpr int kOutSize = 3;

}  // namespace

class LayerTest : public ::testing::Test {
 protected:
  LayerTest()
      : input_tensor_ptr_(AsTensorSharedPtr(xt::random::rand<float>({kDataSize, kInSize}))),
        linear_layer_ptr_(std::make_shared<layer::Linear>(kOutSize)),
        output_tensor_ptr_(linear_layer_ptr_->Call({input_tensor_ptr_})[0]) {}  // y = x W + b

  const TensorSharedPtr input_tensor_ptr_;
  const LayerSharedPtr linear_layer_ptr_;
  const TensorSharedPtr output_tensor_ptr_;
};

TEST_F(LayerTest, CallTest) {
  const xt::xarray<float>::shape_type& actual_input_data_shape = input_tensor_ptr_->data().shape();
  const xt::xarray<float>::shape_type expected_input_data_shape({kDataSize, kInSize});

  const xt::xarray<float>::shape_type& actual_output_data_shape = output_tensor_ptr_->data().shape();
  const xt::xarray<float>::shape_type expected_output_data_shape({kDataSize, kOutSize});

  // Checks that the forward calculation is correct, at least in "shape" level (not in "value" level).
  EXPECT_EQ(actual_input_data_shape, expected_input_data_shape);
  EXPECT_EQ(actual_output_data_shape, expected_output_data_shape);

  // Checks that the input/output tensor pointers are correct.
  EXPECT_EQ(linear_layer_ptr_->input_tensor_ptrs()[0].lock(), input_tensor_ptr_);
  EXPECT_EQ(linear_layer_ptr_->output_tensor_ptrs()[0].lock(), output_tensor_ptr_);
}

TEST_F(LayerTest, ClearGradsTest) {
  output_tensor_ptr_->Backpropagation();

  // Checks that parameter tensors (and input tensors) have their gradient after backpropagation.
  for (const auto& param_name_ptr : linear_layer_ptr_->param_map()) {
    const ParameterSharedPtr param_ptr = param_name_ptr.second;
    EXPECT_TRUE(param_ptr->grad_opt().has_value());
  }
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());

  linear_layer_ptr_->ClearGrads();

  // Checks that parameter tensors don't have (but input tensors still have) their gradient after gradient clearing.
  for (const auto& param_name_ptr : linear_layer_ptr_->param_map()) {
    const ParameterSharedPtr param_ptr = param_name_ptr.second;
    EXPECT_FALSE(param_ptr->grad_opt().has_value());
  }
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());
}

}  // namespace tensorward::core

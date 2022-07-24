#include "tensorward/core/layer.h"

#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "tensorward/core/tensor.h"
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
        output_tensor_ptr_(linear_layer_ptr_->Call({input_tensor_ptr_})[0]) {}

  const TensorSharedPtr input_tensor_ptr_;
  // NOTE: Need to use `shared_ptr<layer::Linear>` instead of `shared_ptr<Layer>` to use `W_name()` and `b_name()`.
  const std::shared_ptr<layer::Linear> linear_layer_ptr_;
  const TensorSharedPtr output_tensor_ptr_;
};

TEST_F(LayerTest, CallTest) {
  const xt::xarray<float>& actual_output_data = output_tensor_ptr_->data();

  // y = x W + b
  const xt::xarray<float>& x_data = input_tensor_ptr_->data();
  const xt::xarray<float>& W_data = linear_layer_ptr_->param_map().at(linear_layer_ptr_->W_name())->data();
  const xt::xarray<float>& b_data = linear_layer_ptr_->param_map().at(linear_layer_ptr_->b_name())->data();
  const xt::xarray<float> expected_output_data = xt::linalg::dot(x_data, W_data) + b_data;

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_data, expected_output_data);

  // Checks that the input/output tensor pointers are correct.
  EXPECT_EQ(linear_layer_ptr_->input_tensor_ptrs()[0].lock(), input_tensor_ptr_);
  EXPECT_EQ(linear_layer_ptr_->output_tensor_ptrs()[0].lock(), output_tensor_ptr_);
}

TEST_F(LayerTest, ClearGradsTest) {
  output_tensor_ptr_->Backpropagation();

  // Checks that parameter tensors (and input tensors) have their gradient after backpropagation.
  EXPECT_TRUE(linear_layer_ptr_->param_map().at(linear_layer_ptr_->W_name())->grad_opt().has_value());
  EXPECT_TRUE(linear_layer_ptr_->param_map().at(linear_layer_ptr_->b_name())->grad_opt().has_value());
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());

  linear_layer_ptr_->ClearGrads();

  // Checks that parameter tensors don't have (but input tensors still have) their gradient after gradient clearing.
  EXPECT_FALSE(linear_layer_ptr_->param_map().at(linear_layer_ptr_->W_name())->grad_opt().has_value());
  EXPECT_FALSE(linear_layer_ptr_->param_map().at(linear_layer_ptr_->b_name())->grad_opt().has_value());
  EXPECT_TRUE(input_tensor_ptr_->grad_opt().has_value());
}

}  // namespace tensorward::core

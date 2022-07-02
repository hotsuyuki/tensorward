#include "tensorward/function/square.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/util/numerical_gradient.h"

namespace tensorward::function {

namespace {

constexpr int kHight = 2;
constexpr int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;

}  // namespace

class SquareTest : public ::testing::Test {
 protected:
  SquareTest()
      : input_data_(xt::random::rand<float>({kHight, kWidth})),
        expected_output_data_(xt::square(input_data_)),
        square_function_ptr_(std::make_shared<Square>()) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr square_function_ptr_;
};

TEST_F(SquareTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = square_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(SquareTest, AnalyticalBackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = square_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = square_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // y = x^2 ---> dy_dx = 2x
  const xt::xarray<float> expected_input_grad = 2.0 * input_data_;

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad);
}

TEST_F(SquareTest, NumericalBackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = square_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = square_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  const std::vector<xt::xarray<float>> expected_input_grads =
      util::NumericalGradient(square_function_ptr_, {input_data_}, kEpsilon);
  ASSERT_EQ(expected_input_grads.size(), 1);

  // Checks that the backward calculation is correct (numerically).
  ASSERT_EQ(actual_input_grads.size(), expected_input_grads.size());
  ASSERT_EQ(actual_input_grads[0].shape(), expected_input_grads[0].shape());
  ASSERT_EQ(actual_input_grads[0].shape(0), kHight);
  ASSERT_EQ(actual_input_grads[0].shape(1), kWidth);
  for (std::size_t i = 0; i < kHight; ++i) {
    for (std::size_t j = 0; j < kWidth; ++j) {
      EXPECT_NEAR(actual_input_grads[0](i, j), expected_input_grads[0](i, j), kEpsilon);
    }
  }
}

TEST_F(SquareTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `square()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = square(input_tensor_ptr);

  // Checks that the output data is correct.
  EXPECT_EQ(output_tensor_ptr->data(), expected_output_data_);

  // Checks that the computational graph is correct.
  //
  // The correct computational graph is:
  //    input_tensors <--- this_function <==> output_tensors
  //
  // The code below checks it with the following order:
  // 1. input_tensors      this_function <--- output_tensors
  // 2. input_tensors <--- this_function      output_tensors
  // 3. input_tensors      this_function ---> output_tensors
  //
  ASSERT_TRUE(output_tensor_ptr->parent_function_ptr());
  const core::FunctionSharedPtr parent_function_ptr = output_tensor_ptr->parent_function_ptr();
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[0], input_tensor_ptr);
  EXPECT_EQ(parent_function_ptr->output_tensor_ptrs()[0].lock(), output_tensor_ptr);
}

}  // namespace tensorward::function

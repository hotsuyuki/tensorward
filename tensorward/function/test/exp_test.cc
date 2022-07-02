#include "tensorward/function/exp.h"

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

class ExpTest : public ::testing::Test {
 protected:
  ExpTest()
      : input_data_(xt::random::rand<float>({kHight, kWidth})),
        expected_output_data_(xt::exp(input_data_)),
        exp_function_ptr_(std::make_shared<Exp>()) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr exp_function_ptr_;
};

TEST_F(ExpTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = exp_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(ExpTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = exp_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = exp_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // y = exp(x) ---> dy_dx = exp(x)
  const xt::xarray<float> expected_input_grad = xt::exp(input_data_);

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad);

  const std::vector<xt::xarray<float>> expected_input_grads_numerical =
      util::NumericalGradient(exp_function_ptr_, {input_data_}, kEpsilon);
  ASSERT_EQ(expected_input_grads_numerical.size(), 1);

  // Checks that the backward calculation is correct (numerically).
  ASSERT_EQ(actual_input_grads.size(), expected_input_grads_numerical.size());
  for (std::size_t n = 0; n < expected_input_grads_numerical.size(); ++n) {
    ASSERT_EQ(actual_input_grads[n].shape(), expected_input_grads_numerical[n].shape());
    ASSERT_EQ(actual_input_grads[n].shape(0), kHight);
    ASSERT_EQ(actual_input_grads[n].shape(1), kWidth);
    for (std::size_t i = 0; i < kHight; ++i) {
      for (std::size_t j = 0; j < kWidth; ++j) {
        EXPECT_NEAR(actual_input_grads[n](i, j), expected_input_grads_numerical[n](i, j), kEpsilon);
      }
    }
  }
}

TEST_F(ExpTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `exp()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = exp(input_tensor_ptr);

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

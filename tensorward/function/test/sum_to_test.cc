#include "tensorward/function/sum_to.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class SumToTest : public ::testing::Test {
 protected:
  SumToTest()
      : input_data_(xt::random::rand<float>({kHeight, kWidth})),
        output_shape_({1, kWidth}),
        expected_output_data_(xt::sum(input_data_, {0}, xt::keep_dims)),
        sum_to_function_ptr_(std::make_shared<SumTo>(output_shape_)) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float>::shape_type output_shape_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr sum_to_function_ptr_;
};

TEST_F(SumToTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = sum_to_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(SumToTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = sum_to_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = sum_to_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  const xt::xarray<float> expected_input_grad = xt::ones_like(input_data_);

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad);
}

TEST_F(SumToTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `sum_to()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = sum_to(input_tensor_ptr, output_shape_);

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

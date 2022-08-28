#include "tensorward/function/relu.h"

#include <gtest/gtest.h>

namespace tensorward::function {

class ReLUTest : public ::testing::Test {
 protected:
  ReLUTest()
      : input_data_(xt::xarray<float>({{4.2, 0.0, -4.2}, {-4.2, 0.0, 4.2}})),
        expected_output_data_(xt::xarray<float>({{4.2, 0.0, 0.0}, {0.0, 0.0, 4.2}})),
        expected_input_grad_(xt::xarray<float>({{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}})),
        relu_function_ptr_(std::make_shared<ReLU>()) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float> expected_output_data_;
  const xt::xarray<float> expected_input_grad_;
  const core::FunctionSharedPtr relu_function_ptr_;
};

TEST_F(ReLUTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = relu_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(ReLUTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = relu_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = relu_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad_);
}

TEST_F(ReLUTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `relu()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = relu(input_tensor_ptr);

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

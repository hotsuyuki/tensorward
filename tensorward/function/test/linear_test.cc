#include "tensorward/function/linear.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kDataSize = 10;
constexpr int kInSize = 2;
constexpr int kOutSize = 3;

}  // namespace

class LinearTest : public ::testing::Test {
 protected:
  LinearTest()
      : input_data0_(xt::random::rand<float>({kDataSize, kInSize})),                        // x
        input_data1_(xt::random::rand<float>({kInSize, kOutSize})),                         // W
        input_data2_(xt::random::rand<float>({1, kOutSize})),                               // b
        expected_output_data_(xt::linalg::dot(input_data0_, input_data1_) + input_data2_),  // y = x W + b
        linear_function_ptr_(std::make_shared<Linear>()) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data1_;
  const xt::xarray<float> input_data2_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr linear_function_ptr_;
};

TEST_F(LinearTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data0_, input_data1_, input_data2_});
  const std::vector<xt::xarray<float>> actual_output_datas = linear_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(LinearTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data0_),
                                                                 core::AsTensorSharedPtr(input_data1_),
                                                                 core::AsTensorSharedPtr(input_data2_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = linear_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = linear_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 3);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  // y = x W + b ---> dL_dx = dL_dy W.T
  const xt::xarray<float> expected_input_grad0 =
      xt::linalg::dot(xt::ones_like(expected_output_data_), xt::transpose(input_data1_));

  // y = x W + b ---> dL_dW = x.T dL_dy
  const xt::xarray<float> expected_input_grad1 =
      xt::linalg::dot(xt::transpose(input_data0_), xt::ones_like(expected_output_data_));

  // y = x W + b ---> dL_db = 1 ... but needs to be summed due to the broadcast in the forward calculation.
  xt::xarray<float> expected_input_grad2 = xt::ones_like(expected_output_data_);
  expected_input_grad2 = xt::sum(expected_input_grad2, {0}, xt::keep_dims);

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad0);
  EXPECT_EQ(actual_input_grads[1], expected_input_grad1);
  EXPECT_EQ(actual_input_grads[2], expected_input_grad2);
}

TEST_F(LinearTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr0 = core::AsTensorSharedPtr(input_data0_);
  const core::TensorSharedPtr input_tensor_ptr1 = core::AsTensorSharedPtr(input_data1_);
  const core::TensorSharedPtr input_tensor_ptr2 = core::AsTensorSharedPtr(input_data2_);

  // `linear()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = linear(input_tensor_ptr0, input_tensor_ptr1, input_tensor_ptr2);

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
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[1], input_tensor_ptr1);
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[2], input_tensor_ptr2);
  EXPECT_EQ(parent_function_ptr->output_tensor_ptrs()[0].lock(), output_tensor_ptr);
}

}  // namespace tensorward::function

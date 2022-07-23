#include "tensorward/function/mean_squared_error.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kNumData = 10;
constexpr int kOutDim = 3;

}  // namespace

class MeanSquaredErrorTest : public ::testing::Test {
 protected:
  MeanSquaredErrorTest()
      : input_data0_(xt::random::rand<float>({kNumData, kOutDim})),
        input_data1_(xt::random::rand<float>({kNumData, kOutDim})),
        expected_output_data_(xt::sum(xt::square(input_data0_ - input_data1_)) / kNumData),
        mean_squared_error_function_ptr_(std::make_shared<MeanSquaredError>()) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data1_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr mean_squared_error_function_ptr_;
};

TEST_F(MeanSquaredErrorTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data0_, input_data1_});
  const std::vector<xt::xarray<float>> actual_output_datas =
      mean_squared_error_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(MeanSquaredErrorTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors(
      {core::AsTensorSharedPtr(input_data0_), core::AsTensorSharedPtr(input_data1_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors =
      mean_squared_error_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads =
      mean_squared_error_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 2);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  // y = sum((x0 - x1)^2) / N = sum((x0 - x1)^2 / N)
  // ---> dy_dx0 = broadcast_to(1.0, x0_shape) * (2(x0 - x1) / N) = 2(x0 - x1) / N
  const xt::xarray<float> expected_input_grad0 = 2.0 * (input_data0_ - input_data1_) / kNumData;

  // y = sum((x0 - x1)^2) / N = sum((x0 - x1)^2 / N)
  // ---> dy_dx1 = broadcast_to(1.0, x1_shape) * (-2(x0 - x1) / N) = -2(x0 - x1) / N
  const xt::xarray<float> expected_input_grad1 = -2.0 * (input_data0_ - input_data1_) / kNumData;

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad0);
  EXPECT_EQ(actual_input_grads[1], expected_input_grad1);
}

TEST_F(MeanSquaredErrorTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr0 = core::AsTensorSharedPtr(input_data0_);
  const core::TensorSharedPtr input_tensor_ptr1 = core::AsTensorSharedPtr(input_data1_);

  // `mean_squared_error()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = mean_squared_error(input_tensor_ptr0, input_tensor_ptr1);

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
  EXPECT_EQ(parent_function_ptr->output_tensor_ptrs()[0].lock(), output_tensor_ptr);
}

}  // namespace tensorward::core

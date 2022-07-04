#include "tensorward/function/sum.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kHight = 2;
constexpr int kWidth = 3;

}  // namespace

class SumTest : public ::testing::Test {
 protected:
  SumTest()
      : input_data_(xt::random::rand<float>({kHight, kWidth})),
        expected_output_data_(xt::sum(input_data_)),
        sum_function_ptr_(std::make_shared<Sum>()) {}

  const xt::xarray<float> input_data_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr sum_function_ptr_;
};

TEST_F(SumTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = sum_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(SumTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = sum_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = sum_function_ptr_->Backward(actual_output_grads);
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

TEST_F(SumTest, BackwardWithAxesButWithoutKeepDimsFlagTest) {
  const xt::xarray<float>::shape_type axes = {1};
  const bool does_keep_dims = false;
  const core::FunctionSharedPtr sum_function_ptr = std::make_shared<Sum>(axes, does_keep_dims);

  // clang-format off
  const xt::xarray<float> input_data({{11, 12, 13},
                                      {21, 22, 23}});
  // clang-format on

  // NOTE: The shape of the expected output data is {2} with 1 dimension, not {2, 1} with 2 dimensions,
  // NOTE: because of "with axes but without keep dims flag".
  const xt::xarray<float> expected_output_data({36, 66});

  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = sum_function_ptr->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_tensors[0]->data(), expected_output_data);
  EXPECT_EQ(actual_output_tensors[0]->data().shape(), xt::xarray<float>::shape_type({2}));
  EXPECT_EQ(actual_output_tensors[0]->data().dimension(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = sum_function_ptr->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data,
  // even though some dimensions were reduced in the forward calculation due to "with axes but without keep dims flag".
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  const xt::xarray<float> expected_input_grad = xt::ones_like(input_data);

  // Checks that the backward calculation is correct (analytically),
  // even though some dimensions were reduced in the forward calculation due to "with axes but without keep dims flag".
  EXPECT_EQ(actual_input_grads[0], expected_input_grad);
}

TEST_F(SumTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `sum()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = sum(input_tensor_ptr);

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

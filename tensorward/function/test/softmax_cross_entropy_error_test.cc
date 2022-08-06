#include "tensorward/function/softmax_cross_entropy_error.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kDataSize = 2;
constexpr int kOutSize = 3;
constexpr float kEpsilon = 1.0e-3;

}  // namespace

class SoftmaxCrossEntropyErrorTest : public ::testing::Test {
 protected:
  SoftmaxCrossEntropyErrorTest()
      : input_data0_(xt::random::rand<float>({kDataSize, kOutSize})),
        input_data0_exp_(xt::exp(input_data0_)),                               // NOTE: Need to construct for softmax.
        input_data0_sum_exp_(xt::sum(input_data0_exp_, {-1}, xt::keep_dims)),  // NOTE: Need to construct for softmax.
        input_data0_softmax_(input_data0_exp_ / input_data0_sum_exp_),               // p
        input_data1_onehot_(xt::xarray<float>({{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}})),  // t (onehot)
        input_data1_non_onehot_(xt::xarray<float>({0.0, 2.0})),                      // t (non-onehot)
        expected_output_data_(-xt::sum(input_data1_onehot_ * xt::log(input_data0_softmax_)) / kDataSize),
        softmax_cross_entropy_error_function_ptr_(std::make_shared<SoftmaxCrossEntropyError>()) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data0_exp_;
  const xt::xarray<float> input_data0_sum_exp_;
  const xt::xarray<float> input_data0_softmax_;
  const xt::xarray<float> input_data1_onehot_;
  const xt::xarray<float> input_data1_non_onehot_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr softmax_cross_entropy_error_function_ptr_;
};

TEST_F(SoftmaxCrossEntropyErrorTest, ForwardTest) {
  // Tests with both onehot label and non-onehot label.
  for (const auto& input_data1 : {input_data1_onehot_, input_data1_non_onehot_}) {
    const std::vector<xt::xarray<float>> actual_input_datas({input_data0_, input_data1});
    const std::vector<xt::xarray<float>> actual_output_datas =
        softmax_cross_entropy_error_function_ptr_->Forward(actual_input_datas);
    ASSERT_EQ(actual_output_datas.size(), 1);

    // Checks that the forward calculation is correct.
    ASSERT_EQ(actual_output_datas[0].shape(), expected_output_data_.shape());
    for (std::size_t i = 0; i < kDataSize; ++i) {
      for (std::size_t j = 0; j < kOutSize; ++j) {
        EXPECT_NEAR(actual_output_datas[0](i, j), expected_output_data_(i, j), kEpsilon);
      }
    }
  }
}

TEST_F(SoftmaxCrossEntropyErrorTest, BackwardTest) {
  // y = cross_entropy_error(softmax(x), t) = cross_entropy_error(p, t) ---> dy_dx = (p - t) / N
  const xt::xarray<float> expected_input_grad0 = (input_data0_softmax_ - input_data1_onehot_) / kDataSize;

  // Tests with both onehot label and non-onehot label.
  for (const auto& input_data1 : {input_data1_onehot_, input_data1_non_onehot_}) {
    // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
    const std::vector<core::TensorSharedPtr> actual_input_tensors(
        {core::AsTensorSharedPtr(input_data0_), core::AsTensorSharedPtr(input_data1)});
    const std::vector<core::TensorSharedPtr> actual_output_tensors =
        softmax_cross_entropy_error_function_ptr_->Call(actual_input_tensors);
    ASSERT_EQ(actual_output_tensors.size(), 1);

    const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
    const std::vector<xt::xarray<float>> actual_input_grads =
        softmax_cross_entropy_error_function_ptr_->Backward(actual_output_grads);
    ASSERT_EQ(actual_input_grads.size(), 2);

    // Checks that the shape of the gradient is the same as the shape of the corresponding data.
    ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
    for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
      EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
    }

    // Checks that the backward calculation is correct (analytically).
    // NOTE: Only checks the gradient of the 1st input, which is `x`, and doesn't check the 2nd input, which is `t`,
    // NOTE: because it doesn't matter since the `t` is not a parameter.
    ASSERT_EQ(actual_input_grads[0].shape(), expected_input_grad0.shape());
    for (std::size_t i = 0; i < kDataSize; ++i) {
      for (std::size_t j = 0; j < kOutSize; ++j) {
        EXPECT_NEAR(actual_input_grads[0](i, j), expected_input_grad0(i, j), kEpsilon);
      }
    }
  }
}

TEST_F(SoftmaxCrossEntropyErrorTest, CallWrapperTest) {
  // Tests with both onehot label and non-onehot label.
  for (const auto& input_data1 : {input_data1_onehot_, input_data1_non_onehot_}) {
    const core::TensorSharedPtr input_tensor_ptr0 = core::AsTensorSharedPtr(input_data0_);
    const core::TensorSharedPtr input_tensor_ptr1 = core::AsTensorSharedPtr(input_data1);

    // `softmax_cross_entropy_error()` is a `Function::Call()` wrapper.
    const core::TensorSharedPtr output_tensor_ptr = softmax_cross_entropy_error(input_tensor_ptr0, input_tensor_ptr1);

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
}

}  // namespace tensorward::core

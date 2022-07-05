#include "tensorward/core/operator/div.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/util/numerical_gradient.h"

namespace tensorward::core {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;
constexpr float kEpsilon = 1.0e-3;
constexpr int kRandomIntLower = -100;
constexpr int kRandomIntUpper = 100;

}  // namespace

class DivTest : public ::testing::Test {
 protected:
  DivTest()
      // NOTE: Need to use integer arrays instead of decimal arrays to avoid numerical error in numerical gradient.
      : input_data0_(xt::random::randint<float>({kHeight, kWidth}, kRandomIntLower, kRandomIntUpper)),
        input_data1_(xt::random::randint<float>({kHeight, kWidth}, kRandomIntLower, kRandomIntUpper)),
        expected_output_data_(input_data0_ / input_data1_),
        div_function_ptr_(std::make_shared<Div>()) {}

  const xt::xarray<float> input_data0_;
  const xt::xarray<float> input_data1_;
  const xt::xarray<float> expected_output_data_;
  const FunctionSharedPtr div_function_ptr_;
};

TEST_F(DivTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data0_, input_data1_});
  const std::vector<xt::xarray<float>> actual_output_datas = div_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(DivTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<TensorSharedPtr> actual_input_tensors(
      {AsTensorSharedPtr(input_data0_), AsTensorSharedPtr(input_data1_)});
  const std::vector<TensorSharedPtr> actual_output_tensors = div_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = div_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 2);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  // y = x0 / x1 = x1^(-1) * x0 ---> dy_dx0 = x1^(-1)
  const xt::xarray<float> expected_input_grad0 = 1.0 / input_data1_;

  // y = x0 / x1 = x0 * x1^(-1) ---> dy_dx1 = -x0 * x1^(-2)
  const xt::xarray<float> expected_input_grad1 = -input_data0_ / xt::square(input_data1_);

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad0);
  EXPECT_EQ(actual_input_grads[1], expected_input_grad1);

  const std::vector<xt::xarray<float>> expected_input_grads_numerical =
      util::NumericalGradient(div_function_ptr_, {input_data0_, input_data1_}, kEpsilon);
  ASSERT_EQ(expected_input_grads_numerical.size(), 2);

  // Checks that the backward calculation is correct (numerically).
  ASSERT_EQ(actual_input_grads.size(), expected_input_grads_numerical.size());
  for (std::size_t n = 0; n < expected_input_grads_numerical.size(); ++n) {
    ASSERT_EQ(actual_input_grads[n].shape(), expected_input_grads_numerical[n].shape());
    ASSERT_EQ(actual_input_grads[n].shape(0), kHeight);
    ASSERT_EQ(actual_input_grads[n].shape(1), kWidth);
    for (std::size_t i = 0; i < kHeight; ++i) {
      for (std::size_t j = 0; j < kWidth; ++j) {
        EXPECT_NEAR(actual_input_grads[n](i, j), expected_input_grads_numerical[n](i, j), kEpsilon);
      }
    }
  }
}

TEST_F(DivTest, BroadcastTest) {
  // Prepares another input data to be broadcasted. The size of the 1st dimension is 1 instead of `kHeight` on purpose.
  const xt::xarray<float> input_data_broadcast(xt::random::rand<float>({1, kWidth}));

  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<TensorSharedPtr> actual_input_tensors(
      {AsTensorSharedPtr(input_data0_), AsTensorSharedPtr(input_data_broadcast)});
  const std::vector<TensorSharedPtr> actual_output_tensors = div_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  // Checks that the shape of the output data is the same as the shape of the input data whose size is larger,
  // which means the broadcast happened correctly.
  EXPECT_TRUE(actual_output_tensors[0]->data().shape() == input_data0_.shape() &&
              actual_output_tensors[0]->data().shape() != input_data_broadcast.shape());

  // Checks that the forward calculation is correct.
  const xt::xarray<float> expected_output_data_broadcast = input_data0_ / input_data_broadcast;
  EXPECT_EQ(actual_output_tensors[0]->data(), expected_output_data_broadcast);

  // The backward calculation.
  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = div_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 2);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  EXPECT_TRUE(actual_input_grads[0].shape() == input_data0_.shape() &&
              actual_input_grads[0].shape() != input_data_broadcast.shape());
  EXPECT_TRUE(actual_input_grads[1].shape() != input_data0_.shape() &&
              actual_input_grads[1].shape() == input_data_broadcast.shape());

  // y = x0 / x1 ---> dy_dx0 = x1^(-1) ... but need to be broadcasted due to the broadcast in the forward calculation.
  xt::xarray<float> expected_input_grad0 = 1.0 / input_data_broadcast;
  expected_input_grad0 = xt::broadcast(expected_input_grad0, input_data0_.shape());

  // y = x0 / x1 ---> dy_dx1 = -x0 * x1^(-2) ... but need to be summed due to the broadcast in the forward calculation.
  xt::xarray<float> expected_input_grad_broadcast = -input_data0_ / xt::square(input_data_broadcast);
  expected_input_grad_broadcast = xt::sum(expected_input_grad_broadcast, {0}, xt::keep_dims);

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad0);
  EXPECT_EQ(actual_input_grads[1], expected_input_grad_broadcast);
}

TEST_F(DivTest, CallWrapperTest) {
  const TensorSharedPtr input_tensor_ptr0 = AsTensorSharedPtr(input_data0_);
  const TensorSharedPtr input_tensor_ptr1 = AsTensorSharedPtr(input_data1_);

  // `div()` is a `Function::Call()` wrapper.
  const TensorSharedPtr output_tensor_ptr = div(input_tensor_ptr0, input_tensor_ptr1);

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
  const FunctionSharedPtr parent_function_ptr = output_tensor_ptr->parent_function_ptr();
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_EQ(parent_function_ptr->input_tensor_ptrs()[1], input_tensor_ptr1);
  EXPECT_EQ(parent_function_ptr->output_tensor_ptrs()[0].lock(), output_tensor_ptr);
}

TEST_F(DivTest, OverloadedOperatorTest) {
  const TensorSharedPtr input_tensor_ptr0 = AsTensorSharedPtr(input_data0_);
  const TensorSharedPtr input_tensor_ptr1 = AsTensorSharedPtr(input_data1_);

  // Tensor pointer / Tensor pointer
  const TensorSharedPtr output_tensor_ptr1 = input_tensor_ptr0 / input_tensor_ptr1;
  EXPECT_EQ(output_tensor_ptr1->data(), expected_output_data_);
  EXPECT_EQ(output_tensor_ptr1->parent_function_ptr()->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_EQ(output_tensor_ptr1->parent_function_ptr()->input_tensor_ptrs()[1], input_tensor_ptr1);

  // Tensor pointer / Array object
  const TensorSharedPtr output_tensor_ptr2 = input_tensor_ptr0 / input_data1_;
  EXPECT_EQ(output_tensor_ptr2->data(), expected_output_data_);
  EXPECT_EQ(output_tensor_ptr2->parent_function_ptr()->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_NE(output_tensor_ptr2->parent_function_ptr()->input_tensor_ptrs()[1], input_tensor_ptr1);

  // Array object / Tensor pointer
  const TensorSharedPtr output_tensor_ptr3 = input_data0_ / input_tensor_ptr1;
  EXPECT_EQ(output_tensor_ptr3->data(), expected_output_data_);
  EXPECT_NE(output_tensor_ptr3->parent_function_ptr()->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_EQ(output_tensor_ptr3->parent_function_ptr()->input_tensor_ptrs()[1], input_tensor_ptr1);

  const float input_scalar0 = input_data0_(0, 0);
  const float input_scalar1 = input_data1_(0, 0);

  // Tensor pointer / Scalar object
  // (Scalar object will be silently converted to Array object)
  const TensorSharedPtr output_tensor_ptr4 = input_tensor_ptr0 / input_scalar1;
  EXPECT_EQ(output_tensor_ptr4->data(), input_data0_ / input_scalar1);
  EXPECT_EQ(output_tensor_ptr4->parent_function_ptr()->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_NE(output_tensor_ptr4->parent_function_ptr()->input_tensor_ptrs()[1], input_tensor_ptr1);

  // Scalar object / Tensor pointer
  // (Scalar object will be silently converted to Array object)
  const TensorSharedPtr output_tensor_ptr5 = input_scalar0 / input_tensor_ptr1;
  EXPECT_EQ(output_tensor_ptr5->data(), input_scalar0 / input_data1_);
  EXPECT_NE(output_tensor_ptr5->parent_function_ptr()->input_tensor_ptrs()[0], input_tensor_ptr0);
  EXPECT_EQ(output_tensor_ptr5->parent_function_ptr()->input_tensor_ptrs()[1], input_tensor_ptr1);
}

}  // namespace tensorward::core


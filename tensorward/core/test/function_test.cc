#include "tensorward/core/function.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/tensor.h"
#include "tensorward/core/operator/add.h"

namespace tensorward::core {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class FunctionTest : public ::testing::Test {
 protected:
  FunctionTest()
      : input_tensor_ptr0_(AsTensorSharedPtr(xt::random::rand<float>({kHeight, kWidth}))),
        input_tensor_ptr1_(AsTensorSharedPtr(xt::random::rand<float>({kHeight, kWidth}))) {}

  const TensorSharedPtr input_tensor_ptr0_;
  const TensorSharedPtr input_tensor_ptr1_;
};

TEST_F(FunctionTest, CallTest) {
  // out = in0 + in1
  const FunctionSharedPtr add_function_ptr = std::make_shared<Add>();
  const std::vector<TensorSharedPtr> output_tensor_ptrs =
      add_function_ptr->Call({input_tensor_ptr0_, input_tensor_ptr1_});
  ASSERT_EQ(output_tensor_ptrs.size(), 1);

  // Checks that the forward calculation is correct.
  const xt::xarray<float>& actual_output_data = output_tensor_ptrs[0]->data();
  const xt::xarray<float> expected_output_data = input_tensor_ptr0_->data() + input_tensor_ptr1_->data();
  EXPECT_EQ(actual_output_data, expected_output_data);

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
  EXPECT_EQ(output_tensor_ptrs[0]->parent_function_ptr(), add_function_ptr);
  EXPECT_EQ(add_function_ptr->input_tensor_ptrs()[0], input_tensor_ptr0_);
  EXPECT_EQ(add_function_ptr->input_tensor_ptrs()[1], input_tensor_ptr1_);
  EXPECT_EQ(add_function_ptr->output_tensor_ptrs()[0].lock(), output_tensor_ptrs[0]);
}

}  // namespace tensorward::core

#include "tensorward/core/function.h"

#include <algorithm>

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/function/square.h"

namespace tensorward::core {

namespace {

constexpr unsigned int kHight = 2;
constexpr unsigned int kWidth = 3;

}  // namespace

class FunctionTest : public ::testing::Test {
 protected:
  FunctionTest()
      : input_tensor_ptr_(std::make_shared<Tensor>(xt::random::rand<float>({kHight, kWidth}))),
        square_function_ptr_(std::make_shared<function::Square>()) {}

  const TensorSharedPtr input_tensor_ptr_;
  const FunctionSharedPtr square_function_ptr_;
};

TEST_F(FunctionTest, CallTest) {
  const TensorSharedPtr output_tensor_ptr = square_function_ptr_->Call(input_tensor_ptr_);

  // Checks that the forward calculation is correct.
  const xt::xarray<float>& actual_output_data = output_tensor_ptr->data();
  xt::xarray<float> expected_output_data(input_tensor_ptr_->data());
  std::for_each(expected_output_data.begin(), expected_output_data.end(), [](float& elem) { elem = elem * elem; });
  EXPECT_EQ(actual_output_data, expected_output_data);

  // Checks that the computational graph is correct.
  //
  // The correct computational graph is:
  //    input_tensor <--- this_function <==> output_tensor
  //
  // The code below checks it with the following order:
  // 1. input_tensor      this_function <--- output_tensor
  // 2. input_tensor <--- this_function      output_tensor
  // 3. input_tensor      this_function ---> output_tensor
  //
  EXPECT_EQ(output_tensor_ptr->parent_function_ptr(), square_function_ptr_);
  EXPECT_EQ(square_function_ptr_->input_tensor_ptr(), input_tensor_ptr_);
  EXPECT_EQ(square_function_ptr_->output_tensor_ptr().lock(), output_tensor_ptr);
}

}  // namespace tensorward::core

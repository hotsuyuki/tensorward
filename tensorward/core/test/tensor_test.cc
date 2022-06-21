#include "tensorward/core/tensor.h"

#include <algorithm>

#include <gtest/gtest.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/function/exp.h"
#include "tensorward/function/square.h"

namespace tensorward::core {

namespace {

constexpr unsigned int kHight = 2;
constexpr unsigned int kWidth = 3;

const float dL_dy(const float y) {
  return std::exp(y);
}

const float dL_dx(const float x) {
  return std::exp(std::powf(x, 2)) * (2.0 * x);
}

}  // namespace

//
// x = x
// y = f(x) = x^2
// L = g(y) = exp(y) = exp(x^2)
//
//   x   ---> f(x) --->   y   ---> g(y) --->   L
// dL/dx <--- * f' <--- dL/dy <--- * g' <--- dL/dL
//           (dy/dx)              (dL/dy)    (1.0)
//
// dL/dL = 1.0
// dL/dy = exp(y)
// dL/dx = exp(x^2) * 2x
//
class TensorTest : public ::testing::Test {
 protected:
  TensorTest()
      : x_tensor_ptr_(std::make_shared<Tensor>(xt::random::rand<float>({kHight, kWidth}))),
        L_tensor_ptr_(function::exp(function::square(x_tensor_ptr_))) {}

  const TensorSharedPtr x_tensor_ptr_;
  const TensorSharedPtr L_tensor_ptr_;
};

TEST_F(TensorTest, GradDirectGetterTest) {
  // Checks that the input tensor doesn't have its gradient and it fails when trying to get the gradient value
  // by the "direct" getter function before backpropagation.
  ASSERT_TRUE(!x_tensor_ptr_->grad_opt().has_value());
  EXPECT_DEATH(x_tensor_ptr_->grad(), "");

  // Checks that the output tensor doesn't have its gradient and it fails when trying to get the gradient value
  // by the "direct" getter function before backpropagation.
  ASSERT_TRUE(!L_tensor_ptr_->grad_opt().has_value());
  EXPECT_DEATH(L_tensor_ptr_->grad(), "");

  L_tensor_ptr_->Backpropagation();

  // Checks that the input tensor has its gradient and the gradient value can be gotten
  // by the "direct" getter function after backpropagation.
  ASSERT_TRUE(x_tensor_ptr_->grad_opt().has_value());
  const xt::xarray<float>& actual_input_grad = x_tensor_ptr_->grad();
  const xt::xarray<float>& expected_input_grad = x_tensor_ptr_->grad_opt().value();
  EXPECT_EQ(actual_input_grad, expected_input_grad);

  // Checks that the output tensor has its gradient and the gradient value can be gotten
  // by the "direct" getter function after backpropagation.
  ASSERT_TRUE(L_tensor_ptr_->grad_opt().has_value());
  const xt::xarray<float>& actual_output_grad = L_tensor_ptr_->grad();
  const xt::xarray<float>& expected_output_grad = L_tensor_ptr_->grad_opt().value();
  EXPECT_EQ(actual_output_grad, expected_output_grad);
}

TEST_F(TensorTest, BackpropagationTest) {
  // `y`: the intermediate tensor between `x` and `L`.
  const TensorSharedPtr y_tensor_ptr = L_tensor_ptr_->parent_function_ptr()->input_tensor_ptr();

  ASSERT_TRUE(!x_tensor_ptr_->grad_opt().has_value());
  ASSERT_TRUE(!y_tensor_ptr->grad_opt().has_value());
  ASSERT_TRUE(!L_tensor_ptr_->grad_opt().has_value());

  L_tensor_ptr_->Backpropagation();

  ASSERT_TRUE(x_tensor_ptr_->grad_opt().has_value());
  ASSERT_TRUE(y_tensor_ptr->grad_opt().has_value());
  ASSERT_TRUE(L_tensor_ptr_->grad_opt().has_value());

  // Checks that the output tensor `L` has correct gradient (1.0) after backpropagation.
  const xt::xarray<float>& actual_L_grad = L_tensor_ptr_->grad();
  const xt::xarray<float> expected_L_grad(xt::ones_like(L_tensor_ptr_->data()));
  EXPECT_EQ(actual_L_grad, expected_L_grad);

  // Checks that the intermediate tensor `y` has correct gradient (dL/dy) after backpropagation.
  const xt::xarray<float>& actual_y_grad = y_tensor_ptr->grad();
  xt::xarray<float> expected_y_grad(y_tensor_ptr->data());
  std::for_each(expected_y_grad.begin(), expected_y_grad.end(), [](float& elem) { elem = dL_dy(elem); });
  EXPECT_EQ(actual_y_grad, expected_y_grad);

  // Checks that the input tensor `x` has correct gradient (dL/dx) after backpropagation.
  const xt::xarray<float>& actual_x_grad = x_tensor_ptr_->grad();
  xt::xarray<float> expected_x_grad(x_tensor_ptr_->data());
  std::for_each(expected_x_grad.begin(), expected_x_grad.end(), [](float& elem) { elem = dL_dx(elem); });
  EXPECT_EQ(actual_x_grad, expected_x_grad);
}

}  // namespace tensorward::core

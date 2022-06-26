#include "tensorward/core/tensor.h"

#include <algorithm>

#include <gtest/gtest.h>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/function.h"
#include "tensorward/function/exp.h"
#include "tensorward/function/square.h"

namespace tensorward::core {

namespace {

constexpr int kHight = 2;
constexpr int kWidth = 3;

const float dL_dy(const float y) {
  return std::exp(y);
}

const float dL_dx(const float x) {
  return std::exp(std::powf(x, 2)) * (2.0 * x);  // NOTE: Need to use `std::powf()` instead of `std::pow()`.
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
      : x_tensor_ptr_(AsTensorSharedPtr(xt::random::rand<float>({kHight, kWidth}))),
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
}

TEST_F(TensorTest, BackpropagationTest) {
  // `y`: the intermediate tensor between `x` and `L`.
  const TensorSharedPtr y_tensor_ptr = L_tensor_ptr_->parent_function_ptr()->input_tensor_ptrs()[0];

  ASSERT_TRUE(!x_tensor_ptr_->grad_opt().has_value());
  ASSERT_TRUE(!y_tensor_ptr->grad_opt().has_value());
  ASSERT_TRUE(!L_tensor_ptr_->grad_opt().has_value());

  constexpr bool kDoesRetainGrad = true;
  L_tensor_ptr_->Backpropagation(kDoesRetainGrad);

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

TEST_F(TensorTest, SetParentFunctionPtrTest) {
  const xt::xarray<float> array_data = xt::random::rand<float>({kHight, kWidth});
  const TensorSharedPtr foo_tensor_ptr = AsTensorSharedPtr(array_data);
  const FunctionSharedPtr bar_function_ptr = std::make_shared<function::Square>();

  foo_tensor_ptr->SetParentFunctionPtr(bar_function_ptr);

  // Checks that the tensor's parent function is set correctly, and the tensor's generation is 1 older than
  // the parent function's generation.
  EXPECT_EQ(foo_tensor_ptr->parent_function_ptr(), bar_function_ptr);
  EXPECT_EQ(foo_tensor_ptr->generation(), bar_function_ptr->generation() + 1);
}

TEST_F(TensorTest, AsTensorSharedPtrTest) {
  const xt::xarray<float> array_data = xt::random::rand<float>({kHight, kWidth});

  const std::string foo = "foo";
  const TensorSharedPtr foo_tensor_ptr = AsTensorSharedPtr(array_data, foo);

  // Checks that the Tensor instance constructed from an array can be accessed via the returned shared pointer.
  EXPECT_EQ(foo_tensor_ptr.use_count(), 1);
  EXPECT_EQ(foo_tensor_ptr->data(), array_data);
  EXPECT_EQ(foo_tensor_ptr->grad_opt(), std::nullopt);
  EXPECT_EQ(foo_tensor_ptr->name(), foo);
  EXPECT_EQ(foo_tensor_ptr->parent_function_ptr(), nullptr);
  EXPECT_EQ(foo_tensor_ptr->generation(), 0);

  const float scalar_data_0D = array_data(0, 0);
  const xt::xarray<float> scalar_data_1D({scalar_data_0D});
  ASSERT_EQ(scalar_data_1D.dimension(), 1);
  ASSERT_EQ(scalar_data_1D(0), scalar_data_0D);

  const std::string bar = "bar";
  const TensorSharedPtr bar_tensor_ptr = AsTensorSharedPtr(scalar_data_0D, bar);

  // Checks that the Tensor instance constructed from a "0-D" scalar can be accessed via the returned shared pointer,
  // and the data of that Tensor instance is "1-D" scalar.
  EXPECT_EQ(bar_tensor_ptr.use_count(), 1);
  EXPECT_EQ(bar_tensor_ptr->data(), scalar_data_1D);
  EXPECT_EQ(bar_tensor_ptr->grad_opt(), std::nullopt);
  EXPECT_EQ(bar_tensor_ptr->name(), bar);
  EXPECT_EQ(bar_tensor_ptr->parent_function_ptr(), nullptr);
  EXPECT_EQ(bar_tensor_ptr->generation(), 0);
}

}  // namespace tensorward::core

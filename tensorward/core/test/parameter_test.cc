#include "tensorward/core/parameter.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/tensor.h"
#include "tensorward/core/operator/add.h"
#include "tensorward/core/operator/div.h"
#include "tensorward/core/operator/mul.h"
#include "tensorward/core/operator/sub.h"
#include "tensorward/function/mean_squared_error.h"

namespace tensorward::core {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class ParameterTest : public ::testing::Test {
 protected:
  ParameterTest()
      : tensor_ptr_(AsTensorSharedPtr(xt::random::rand<float>({kHeight, kWidth}))),
        parameter_ptr_(AsParameterSharedPtr(xt::random::rand<float>({kHeight, kWidth}))) {}

  const TensorSharedPtr tensor_ptr_;
  const ParameterSharedPtr parameter_ptr_;
};

TEST_F(ParameterTest, TypeidTest) {
  const auto& tensor_obj = *tensor_ptr_;
  const auto& parameter_obj = *parameter_ptr_;

  // Checks that `typeid()` can distinguish between the `Tensor` and `Parameter`, which is derived from `Tensor`.
  EXPECT_EQ(typeid(tensor_obj), typeid(Tensor));
  EXPECT_EQ(typeid(parameter_obj), typeid(Parameter));

  const auto& tensor_add_parameter_obj = *(tensor_ptr_ + parameter_ptr_);
  const auto& tensor_sub_parameter_obj = *(tensor_ptr_ - parameter_ptr_);
  const auto& tensor_mul_parameter_obj = *(tensor_ptr_ * parameter_ptr_);
  const auto& tensor_div_parameter_obj = *(tensor_ptr_ / parameter_ptr_);
  const auto& tensor_matmul_parameter_obj = *(function::mean_squared_error(tensor_ptr_, parameter_ptr_));

  // Checks that the calculation result between `Tensor` class and `Parameter` class is `Tensor` class.
  EXPECT_EQ(typeid(tensor_add_parameter_obj), typeid(Tensor));
  EXPECT_EQ(typeid(tensor_sub_parameter_obj), typeid(Tensor));
  EXPECT_EQ(typeid(tensor_mul_parameter_obj), typeid(Tensor));
  EXPECT_EQ(typeid(tensor_div_parameter_obj), typeid(Tensor));
  EXPECT_EQ(typeid(tensor_matmul_parameter_obj), typeid(Tensor));
}

TEST_F(ParameterTest, AsTensorSharedPtrTest) {
  const xt::xarray<float> array_data = xt::random::rand<float>({kHeight, kWidth});
  const std::string foo_name = "foo";
  const ParameterSharedPtr foo_parameter_ptr = AsParameterSharedPtr(array_data, foo_name);

  // Checks that the Tensor instance constructed from an array can be accessed via the returned shared pointer.
  EXPECT_EQ(foo_parameter_ptr.use_count(), 1);
  EXPECT_EQ(foo_parameter_ptr->data(), array_data);
  EXPECT_EQ(foo_parameter_ptr->data().dimension(), 2);
  EXPECT_EQ(foo_parameter_ptr->data().size(), kHeight * kWidth);
  EXPECT_EQ(foo_parameter_ptr->grad_opt(), std::nullopt);
  EXPECT_EQ(foo_parameter_ptr->name(), foo_name);
  EXPECT_EQ(foo_parameter_ptr->parent_function_ptr(), nullptr);
  EXPECT_EQ(foo_parameter_ptr->generation(), 0);

  const float scalar_data = array_data(0, 0);
  const std::string bar_name = "bar";
  const ParameterSharedPtr bar_parameter_ptr = AsParameterSharedPtr(scalar_data, bar_name);

  // Checks that the Tensor instance constructed from a "0-D" scalar can be accessed via the returned shared pointer.
  EXPECT_EQ(bar_parameter_ptr.use_count(), 1);
  EXPECT_EQ(bar_parameter_ptr->data(), xt::xarray<float>(scalar_data));
  EXPECT_EQ(bar_parameter_ptr->data().dimension(), 0);
  EXPECT_EQ(bar_parameter_ptr->data().size(), 1);
  EXPECT_EQ(bar_parameter_ptr->grad_opt(), std::nullopt);
  EXPECT_EQ(bar_parameter_ptr->name(), bar_name);
  EXPECT_EQ(bar_parameter_ptr->parent_function_ptr(), nullptr);
  EXPECT_EQ(bar_parameter_ptr->generation(), 0);
}

}  // namespace tensorward::core

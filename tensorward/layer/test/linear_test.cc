#include "tensorward/layer/linear.h"

#include <gtest/gtest.h>
#include <xtensor-blas/xlinalg.hpp>

namespace tensorward::layer {

namespace {

constexpr int kDataSize = 10;
constexpr int kInSize = 2;
constexpr int kOutSize = 3;

}  // namespace

class LinearTest : public ::testing::Test {
 protected:
  LinearTest()
      : input_tensor_ptr_(core::AsTensorSharedPtr(xt::random::rand<float>({kDataSize, kInSize}))) {}

  const core::TensorSharedPtr input_tensor_ptr_;
};

TEST_F(LinearTest, ForwardTest) {
  // NOTE: Need to use `shared_ptr<layer::Linear>` instead of `shared_ptr<Layer>` in order to use member functions
  // NOTE: that are defined only in the derived class, which is `layer::Linear`, e.g. `W_name()` and `b_name()`.

  // With bias "b".
  {
    const bool does_use_bias = true;
    const std::shared_ptr<layer::Linear> linear_layer_ptr = std::make_shared<layer::Linear>(kOutSize, does_use_bias);

    const std::vector<core::TensorSharedPtr> actual_input_tensor_ptrs({input_tensor_ptr_});
    const std::vector<core::TensorSharedPtr> actual_output_tensor_ptrs =
        linear_layer_ptr->Forward(actual_input_tensor_ptrs);
    ASSERT_EQ(actual_output_tensor_ptrs.size(), 1);
    const xt::xarray<float>& actual_output_data = actual_output_tensor_ptrs[0]->data();

    ASSERT_EQ(linear_layer_ptr->param_map().size(), 2);

    // y = x W + b
    const xt::xarray<float>& x_data = input_tensor_ptr_->data();
    const xt::xarray<float>& W_data = linear_layer_ptr->param_map().at(linear_layer_ptr->W_name())->data();
    const xt::xarray<float>& b_data = linear_layer_ptr->param_map().at(linear_layer_ptr->b_name())->data();
    const xt::xarray<float> expected_output_data = xt::linalg::dot(x_data, W_data) + b_data;

    // Checks that the forward calculation is correct.
    EXPECT_EQ(actual_output_data, expected_output_data);
  }

  // Without bias "b".
  {
    const bool does_use_bias = false;
    const std::shared_ptr<layer::Linear> linear_layer_ptr = std::make_shared<layer::Linear>(kOutSize, does_use_bias);

    const std::vector<core::TensorSharedPtr> actual_input_tensor_ptrs({input_tensor_ptr_});
    const std::vector<core::TensorSharedPtr> actual_output_tensor_ptrs =
        linear_layer_ptr->Forward(actual_input_tensor_ptrs);
    ASSERT_EQ(actual_output_tensor_ptrs.size(), 1);
    const xt::xarray<float>& actual_output_data = actual_output_tensor_ptrs[0]->data();

    ASSERT_EQ(linear_layer_ptr->param_map().size(), 1);

    // y = x W
    const xt::xarray<float>& x_data = input_tensor_ptr_->data();
    const xt::xarray<float>& W_data = linear_layer_ptr->param_map().at(linear_layer_ptr->W_name())->data();
    const xt::xarray<float> expected_output_data = xt::linalg::dot(x_data, W_data);

    // Checks that the forward calculation is correct.
    EXPECT_EQ(actual_output_data, expected_output_data);
  }
}

}  // namespace tensorward::layer

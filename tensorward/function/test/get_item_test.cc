#include "tensorward/function/get_item.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

namespace tensorward::function {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class GetItemTest : public ::testing::Test {
 protected:
  GetItemTest()
      : input_data_(xt::random::rand<float>({kHeight, kWidth})),
        indices_({{0, 0}, {0, 1}, {0, 2}, {0, 0}, {0, 1}, {0, 2}}),  // Extracts the 1st row twice.
        expected_output_data_(xt::index_view(input_data_, indices_)),
        get_item_function_ptr_(std::make_shared<GetItem>(indices_)) {}

  const xt::xarray<float> input_data_;
  const std::vector<xt::xindex> indices_;
  const xt::xarray<float> expected_output_data_;
  const core::FunctionSharedPtr get_item_function_ptr_;
};

TEST_F(GetItemTest, ForwardTest) {
  const std::vector<xt::xarray<float>> actual_input_datas({input_data_});
  const std::vector<xt::xarray<float>> actual_output_datas = get_item_function_ptr_->Forward(actual_input_datas);
  ASSERT_EQ(actual_output_datas.size(), 1);

  // Checks that the forward calculation is correct.
  EXPECT_EQ(actual_output_datas[0], expected_output_data_);
}

TEST_F(GetItemTest, BackwardTest) {
  // NOTE: Need to use `Call()` instead of `Forward()` in order to create the computational graph for `Backward()`.
  const std::vector<core::TensorSharedPtr> actual_input_tensors({core::AsTensorSharedPtr(input_data_)});
  const std::vector<core::TensorSharedPtr> actual_output_tensors = get_item_function_ptr_->Call(actual_input_tensors);
  ASSERT_EQ(actual_output_tensors.size(), 1);

  const std::vector<xt::xarray<float>> actual_output_grads({xt::ones_like(actual_output_tensors[0]->data())});
  const std::vector<xt::xarray<float>> actual_input_grads = get_item_function_ptr_->Backward(actual_output_grads);
  ASSERT_EQ(actual_input_grads.size(), 1);

  // Checks that the shape of the gradient is the same as the shape of the corresponding data.
  ASSERT_EQ(actual_input_grads.size(), actual_input_tensors.size());
  for (std::size_t i = 0; i < actual_input_grads.size(); ++i) {
    EXPECT_EQ(actual_input_grads[i].shape(), actual_input_tensors[i]->data().shape());
  }

  // clang-format off
  // Because we extracted the 1st row twice, the 1st row of the expected input gradient is all 2.0 (= 1.0 * 2).
  const xt::xarray<float> expected_input_grad({{2.0, 2.0, 2.0},
                                               {0.0, 0.0, 0.0}});
  // clang-format on

  // Checks that the backward calculation is correct (analytically).
  EXPECT_EQ(actual_input_grads[0], expected_input_grad);
}

TEST_F(GetItemTest, CallWrapperTest) {
  const core::TensorSharedPtr input_tensor_ptr = core::AsTensorSharedPtr(input_data_);

  // `get_item()` is a `Function::Call()` wrapper.
  const core::TensorSharedPtr output_tensor_ptr = get_item(input_tensor_ptr, indices_);

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

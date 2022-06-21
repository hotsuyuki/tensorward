#include "tensorward/core/tensor.h"

#include <vector>

#include <xtensor/xbuilder.hpp>

#include "tensorward/core/function.h"

namespace tensorward::core {

void Tensor::Backpropagation() {
  // Sets the gradient as a tensor of ones if the gradient is none (e.g. loss function output).
  if (!grad_opt_.has_value()) {
    grad_opt_ = xt::ones_like(data_);
  }

  // If there doesn't exit a parent function, then it means this tensor is created by an user (not by a function).
  // So, no need to continue the backward calculation.
  if (!parent_function_ptr_) {
    return;
  }

  std::vector<FunctionSharedPtr> parent_function_ptrs({parent_function_ptr_});
  while (!parent_function_ptrs.empty()) {
    //
    // input_tensor          parent_function           output_tensor
    //    dL_dx      <---  Function::Backward()  <---      dL_dy
    //
    const FunctionSharedPtr parent_function_ptr = parent_function_ptrs.back();
    const TensorWeakPtr output_tensor_ptr = parent_function_ptr->output_tensor_ptr();
    const TensorSharedPtr input_tensor_ptr = parent_function_ptr->input_tensor_ptr();

    const xt::xarray<float>& dL_dy = output_tensor_ptr.lock()->grad();
    const xt::xarray<float> dL_dx = parent_function_ptr->Backward(dL_dy);
    input_tensor_ptr->SetGradOpt(dL_dx);

    parent_function_ptrs.pop_back();
    if (input_tensor_ptr->parent_function_ptr()) {
      parent_function_ptrs.push_back(input_tensor_ptr->parent_function_ptr());
    }
  }
}

}  // namespace tensorward::core

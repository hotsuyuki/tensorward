#include "tensorward/core/function.h"

#include <cassert>

#include "tensorward/core/config.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

const std::vector<TensorSharedPtr> Function::Call(const std::vector<TensorSharedPtr>& input_tensor_ptrs) {
  assert(input_tensor_ptrs.size() == num_inputs_);

  // Performs the forward calculation, and creates an output tensor (dynamically in heap memory so that it's accessible
  // even after exiting this scope).
  std::vector<xt::xarray<float>> xs;
  xs.reserve(input_tensor_ptrs.size());
  for (const auto& input_tensor_ptr : input_tensor_ptrs) {
    xs.push_back(input_tensor_ptr->data());
  }
  const std::vector<xt::xarray<float>> ys = Forward(xs);
  std::vector<TensorSharedPtr> output_tensor_ptrs;
  output_tensor_ptrs.reserve(ys.size());
  for (const auto& y : ys) {
    output_tensor_ptrs.push_back(AsTensorSharedPtr(y));
  }

  if (Config::instance().config_value(Config::kDoesEnableBackpropagation)) {
    // Sets the function generation as the max generation among the input tensors.
    const auto max_generation_input_tensor_ptr_itr =
        std::max_element(input_tensor_ptrs.cbegin(), input_tensor_ptrs.cend(),
                         [](const TensorSharedPtr lhs_ptr, const TensorSharedPtr rhs_ptr) {
                           return lhs_ptr->generation() < rhs_ptr->generation();
                         });
    generation_ = (*max_generation_input_tensor_ptr_itr)->generation();

    // TODO: Refactor this code block so that we don't need the temporary variable `output_tensor_weak_ptrs`.
    // Grows the computational graph with Define-by-Run schema.
    std::vector<TensorWeakPtr> output_tensor_weak_ptrs;
    output_tensor_weak_ptrs.reserve(output_tensor_ptrs.size());
    for (const auto& output_tensor_ptr : output_tensor_ptrs) {
      // Converts from "shared" pointers to "weak" pointers.
      output_tensor_weak_ptrs.push_back(output_tensor_ptr);

      const FunctionSharedPtr this_function_ptr = shared_from_this();
      output_tensor_ptr->SetParentFunctionPtr(this_function_ptr);  // input_tensors     this_function <-- output_tensors
    }
    // clang-format off
    input_tensor_ptrs_ = input_tensor_ptrs;                        // input_tensors <-- this_function <-- output_tensors
    output_tensor_ptrs_ = output_tensor_weak_ptrs;                 // input_tensors <-- this_function <=> output_tensors
    // clang-format on
  }

  assert(output_tensor_ptrs.size() == num_outputs_);
  return output_tensor_ptrs;
}

}  // namespace tensorward::core

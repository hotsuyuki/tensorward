#include "tensorward/core/tensor.h"

#include <list>
#include <set>
#include <vector>

#include <xtensor/xbuilder.hpp>

#include "tensorward/core/function.h"

namespace tensorward::core {

void Tensor::Backpropagation(const bool does_retain_grad /* = false */) {
  // Sets the gradient as a tensor of ones if the gradient is none (e.g. loss function output).
  if (!grad_opt_.has_value()) {
    grad_opt_ = xt::ones_like(data_);
  }

  // If there doesn't exit a parent function, then it means this tensor is created by an user (not by a function).
  // So no need to continue the backward calculation.
  if (!parent_function_ptr_) {
    return;
  }

  // Data structure for the backward queue.
  // TODO: Make it more efficient than `std::list` + `std::set` (maybe by `std::priority_queue` + `std::set`)
  std::list<FunctionSharedPtr> parent_function_ptrs_list;
  std::set<FunctionSharedPtr> parent_function_ptrs_history;

  const auto append_parent_function_ptr_if_unique =
      [&parent_function_ptrs_list, &parent_function_ptrs_history](const FunctionSharedPtr parent_function_ptr) {
        if (parent_function_ptrs_history.find(parent_function_ptr) == parent_function_ptrs_history.end()) {
          parent_function_ptrs_history.insert(parent_function_ptr);
          parent_function_ptrs_list.push_back(parent_function_ptr);
          parent_function_ptrs_list.sort([](const FunctionSharedPtr lhs_ptr, const FunctionSharedPtr rhs_ptr) {
            return lhs_ptr->generation() < rhs_ptr->generation();
          });
        }
      };

  // Appends the (first) parent function of this tensor into the backward queue.
  append_parent_function_ptr_if_unique(parent_function_ptr_);

  while (!parent_function_ptrs_list.empty()) {
    // Because the backward queue (the list) is sorted in ascending order by function generation,
    // the max generation function is in the last of the backward queue (the list).
    const FunctionSharedPtr parent_function_ptr = parent_function_ptrs_list.back();
    parent_function_ptrs_list.pop_back();

    const std::vector<TensorWeakPtr>& output_tensor_ptrs = parent_function_ptr->output_tensor_ptrs();
    const std::vector<TensorSharedPtr>& input_tensor_ptrs = parent_function_ptr->input_tensor_ptrs();
    assert(output_tensor_ptrs.size() == parent_function_ptr->num_outputs());
    assert(input_tensor_ptrs.size() == parent_function_ptr->num_inputs());

    std::vector<xt::xarray<float>> dL_dys;
    dL_dys.reserve(output_tensor_ptrs.size());
    for (const auto& output_tensor_ptr : output_tensor_ptrs) {
      dL_dys.push_back(output_tensor_ptr.lock()->grad());
    }

    //
    // input_tensor          parent_function           output_tensor
    //    dL_dx      <---  Function::Backward()  <---      dL_dy
    //
    assert(dL_dys.size() == output_tensor_ptrs.size());
    const std::vector<xt::xarray<float>> dL_dxs = parent_function_ptr->Backward(dL_dys);
    assert(dL_dxs.size() == input_tensor_ptrs.size());

    for (std::size_t i = 0; i < dL_dxs.size(); ++i) {
      if (!input_tensor_ptrs[i]->grad_opt().has_value()) {
        input_tensor_ptrs[i]->SetGradOpt(dL_dxs[i]);
      } else {
        input_tensor_ptrs[i]->SetGradOpt(input_tensor_ptrs[i]->grad() + dL_dxs[i]);
      }

      // If the parent function exists and hasn't been appended before, then appends it into the backward queue.
      if (input_tensor_ptrs[i]->parent_function_ptr()) {
        append_parent_function_ptr_if_unique(input_tensor_ptrs[i]->parent_function_ptr());
      }
    }

    // Clears the gradient of the output tensors if it's not necessary anymore.
    // (e.g. In most cases, we don't care about the gradient of the middle and last tensors in the computational graph.)
    if (!does_retain_grad) {
      for (const auto& output_tensor_ptr : output_tensor_ptrs) {
        output_tensor_ptr.lock()->ClearGrad();
      }
    }
  }
}

void Tensor::SetParentFunctionPtr(const FunctionSharedPtr parent_function_ptr) {
  parent_function_ptr_ = parent_function_ptr;
  generation_ = parent_function_ptr->generation() + 1;
}

const TensorSharedPtr AsTensorSharedPtr(const xt::xarray<float>& data, const std::string& name /* = "" */) {
  return std::make_shared<Tensor>(data, name);
}

}  // namespace tensorward::core

#include "tensorward/core/function.h"

#include "tensorward/core/tensor.h"

namespace tensorward::core {

const TensorSharedPtr Function::Call(const TensorSharedPtr input_tensor_ptr) {
  // Performs the forward calculation, and creates an output tensor (dynamically in heap memory so that it's accessible
  // even after exiting this scope).
  const xt::xarray<float>& x = input_tensor_ptr->data();
  const xt::xarray<float> y = Forward(x);
  const TensorSharedPtr output_tensor_ptr = std::make_shared<Tensor>(y);

  // Grows the computational graph with Define-by-Run schema.
  output_tensor_ptr->SetParentFunctionPtr(shared_from_this());  // input_tensor      this_function <--- output_tensor
  input_tensor_ptr_ = input_tensor_ptr;                         // input_tensor <--- this_function <--- output_tensor
  output_tensor_ptr_ = output_tensor_ptr;  // This's weak_ptr.  // input_tensor <--- this_function <==> output_tensor

  return output_tensor_ptr;
}

}  // namespace tensorward::core


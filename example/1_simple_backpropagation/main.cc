#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "tensorward/core.h"
#include "tensorward/function.h"

#define DEBUG_PRINT(var) std::cout << #var << " = " << std::endl << var << std::endl;

int main(int argc, char* argv[]) {
  // in = 0.5
  const xt::xarray<float> input_data({0.5});
  const tensorward::core::TensorSharedPtr input_tensor_ptr = std::make_shared<tensorward::core::Tensor>(input_data);

  // out = {exp(in^2)}^2
  const tensorward::core::TensorSharedPtr output_tensor_ptr =
      tensorward::function::square(tensorward::function::exp(tensorward::function::square(input_tensor_ptr)));

  DEBUG_PRINT(input_tensor_ptr->data());  // Should print "{ 0.5 }"
  std::cout << std::endl;
  DEBUG_PRINT(output_tensor_ptr->data());  // Should print "{ 1.648721 }"
  std::cout << std::endl;

  output_tensor_ptr->Backpropagation();

  DEBUG_PRINT(input_tensor_ptr->grad());  // Should print "{ 3.297443 }"
  std::cout << std::endl;
  DEBUG_PRINT(output_tensor_ptr->grad());  // Should print "{ 1.0 }"
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

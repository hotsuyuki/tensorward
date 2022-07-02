#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "tensorward/core.h"
#include "tensorward/function.h"

#define DEBUG_PRINT(var) std::cout << #var << " = " << var << std::endl;

namespace tw = tensorward::core;
namespace F = tensorward::function;

int main(int argc, char* argv[]) {
  const tw::TensorSharedPtr x0_ptr = tw::AsTensorSharedPtr(1.0, "x0");
  const tw::TensorSharedPtr x1_ptr = tw::AsTensorSharedPtr(1.0, "x1");

  // Matyas function: z = 0.26 * (x0^2 + x1^2) - 0.48 * x0 * x1
  const tw::TensorSharedPtr y_ptr = 0.26 * (F::square(x0_ptr) + F::square(x1_ptr)) - 0.48 * x0_ptr * x1_ptr;

  y_ptr->Backpropagation();

  DEBUG_PRINT(y_ptr);  // It should print "data: 0.04"
  std::cout << std::endl;

  DEBUG_PRINT(x0_ptr);  // It should print "data: 1.0, grad: 0.04"
  std::cout << std::endl;

  DEBUG_PRINT(x1_ptr);  // It should print "data: 1.0, grad: 0.04"
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

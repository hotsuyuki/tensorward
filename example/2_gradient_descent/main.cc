#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "tensorward/core.h"
#include "tensorward/function.h"

#define DEBUG_PRINT(var) std::cout << #var << " = " << var << std::endl;

namespace tw = tensorward::core;
namespace F = tensorward::function;

int main(int argc, char* argv[]) {
  const float learning_rate = 0.01;
  const std::size_t iterations = 200;

  // x = 2.0
  const tw::TensorSharedPtr x_ptr = tw::AsTensorSharedPtr(2.0, "x");

  for (std::size_t i = 0; i < iterations; ++i) {
    if (i % 10 == 0) {
      DEBUG_PRINT(i);
      DEBUG_PRINT(x_ptr);
      std::cout << std::endl;
    }

    // y = x^4 - 2 * x^2
    const tw::TensorSharedPtr y_ptr = F::pow(x_ptr, 4) - 2.0 * F::square(x_ptr);

    x_ptr->ClearGrad();
    y_ptr->Backpropagation();

    // x <-- x - lr * dy/dx
    x_ptr->SeData(x_ptr->data() - learning_rate * x_ptr->grad());
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  DEBUG_PRINT(x_ptr);  // It should print "data: 1.000001, grad: SOME_SMALL_VALUE"
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

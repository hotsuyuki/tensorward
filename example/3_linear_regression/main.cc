#include <cassert>
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/core.h"
#include "tensorward/function.h"

#define DEBUG_PRINT(var) std::cout << #var << " = " << var << std::endl;

namespace tw = tensorward::core;
namespace F = tensorward::function;

int main(int argc, char* argv[]) {
  constexpr int kDataSize = 100;
  constexpr int kInSize = 1;
  constexpr int kOutSize = 1;

  constexpr float kSlope = 2.0;
  constexpr float kIntercept = 5.0;

  constexpr float kLearningRate = 0.1;
  constexpr std::size_t kIterations = 100;

  // Toy dataset: y = slope * x + intercept + noise
  xt::random::seed(0);
  const xt::xarray<float> x_data = xt::random::rand<float>({kDataSize, kInSize});
  const xt::xarray<float> y_data = (kSlope * x_data + kIntercept) + xt::random::rand<float>({kDataSize, kOutSize});

  // Variables
  const tw::TensorSharedPtr x_ptr = tw::AsTensorSharedPtr(x_data, "x");
  const tw::TensorSharedPtr y_ptr = tw::AsTensorSharedPtr(y_data, "y");

  // Parameters
  const tw::TensorSharedPtr W_ptr = tw::AsTensorSharedPtr(xt::zeros<float>({kInSize, kOutSize}), "W");
  const tw::TensorSharedPtr b_ptr = tw::AsTensorSharedPtr(xt::zeros<float>({kOutSize}), "b");

  for (std::size_t i = 0; i < kIterations; ++i) {
    // Prediction
    tw::TensorSharedPtr y_pred_ptr;
    y_pred_ptr = F::linear(x_ptr, W_ptr, b_ptr);

    // Loss
    const tw::TensorSharedPtr loss_ptr = F::mean_squared_error(y_ptr, y_pred_ptr);

    // Backpropagation
    W_ptr->ClearGrad();
    b_ptr->ClearGrad();
    loss_ptr->Backpropagation();

    // Parameter updating
    W_ptr->SeData(W_ptr->data() - kLearningRate * W_ptr->grad());
    b_ptr->SeData(b_ptr->data() - kLearningRate * b_ptr->grad());

    if (i % (kIterations / 10) == 0) {
      DEBUG_PRINT(i);
      DEBUG_PRINT(loss_ptr);
      std::cout << std::endl;
    }
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  DEBUG_PRINT(W_ptr);  // It should print "data: {{ SOME_VALUE_NEAR_kSlope }}, grad: {{ SOME_SMALL_VALUE }}"
  std::cout << std::endl;

  DEBUG_PRINT(b_ptr);  // It should print "data: { SOME_VALUE_NEAR_kIntercept }, grad: { SOME_SMALL_VALUE }"
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

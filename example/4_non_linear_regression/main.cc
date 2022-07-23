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
  constexpr int kHideSize = 10;
  constexpr int kOutSize = 1;

  constexpr float kLearningRate = 0.2;
  constexpr std::size_t kIterations = 10000;

  // Toy dataset: y = sin(2pi * x) + noise
  xt::random::seed(0);
  const xt::xarray<float> x_data = xt::random::rand<float>({kDataSize, kInSize});
  const xt::xarray<float> y_data = (xt::sin(2.0 * M_PI * x_data)) + xt::random::rand<float>({kDataSize, kOutSize});

  // Variables
  const tw::TensorSharedPtr x_ptr = tw::AsTensorSharedPtr(x_data, "x");
  const tw::TensorSharedPtr y_ptr = tw::AsTensorSharedPtr(y_data, "y");

  // Parameters
  const tw::TensorSharedPtr W1_ptr = tw::AsTensorSharedPtr(0.01 * xt::random::rand<float>({kInSize, kHideSize}), "W1");
  const tw::TensorSharedPtr b1_ptr = tw::AsTensorSharedPtr(xt::zeros<float>({kHideSize}), "b1");
  const tw::TensorSharedPtr W2_ptr = tw::AsTensorSharedPtr(0.01 * xt::random::rand<float>({kHideSize, kOutSize}), "W2");
  const tw::TensorSharedPtr b2_ptr = tw::AsTensorSharedPtr(xt::zeros<float>({kOutSize}), "b2");

  for (std::size_t i = 0; i < kIterations; ++i) {
    // Prediction
    tw::TensorSharedPtr y_pred_ptr;
    y_pred_ptr = F::linear(x_ptr, W1_ptr, b1_ptr);
    y_pred_ptr = F::sigmoid(y_pred_ptr);
    y_pred_ptr = F::linear(y_pred_ptr, W2_ptr, b2_ptr);

    // Loss
    const tw::TensorSharedPtr loss_ptr = F::mean_squared_error(y_ptr, y_pred_ptr);

    // Backpropagation
    W1_ptr->ClearGrad();
    b1_ptr->ClearGrad();
    W2_ptr->ClearGrad();
    b2_ptr->ClearGrad();
    loss_ptr->Backpropagation();

    // Parameter updating
    W1_ptr->SeData(W1_ptr->data() - kLearningRate * W1_ptr->grad());
    b1_ptr->SeData(b1_ptr->data() - kLearningRate * b1_ptr->grad());
    W2_ptr->SeData(W2_ptr->data() - kLearningRate * W2_ptr->grad());
    b2_ptr->SeData(b2_ptr->data() - kLearningRate * b2_ptr->grad());

    if (i % (kIterations / 10) == 0) {
      DEBUG_PRINT(i);
      DEBUG_PRINT(loss_ptr);
      std::cout << std::endl;
    }
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  DEBUG_PRINT(W1_ptr);
  std::cout << std::endl;

  DEBUG_PRINT(b1_ptr);
  std::cout << std::endl;

  DEBUG_PRINT(W2_ptr);
  std::cout << std::endl;

  DEBUG_PRINT(b2_ptr);
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

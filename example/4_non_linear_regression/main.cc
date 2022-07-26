#include <cassert>
#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "tensorward/core.h"
#include "tensorward/function.h"
#include "tensorward/model.h"
#include "tensorward/optimizer.h"

#define DEBUG_PRINT(var) std::cout << #var << " = " << var << std::endl;

namespace tw = tensorward::core;
namespace F = tensorward::function;
namespace M = tensorward::model;
namespace O = tensorward::optimizer;

int main(int argc, char* argv[]) {
  constexpr int kDataSize = 100;
  constexpr int kInSize = 1;
  constexpr int kHiddenSize = 10;
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

  // Model
  M::MultiLayerPerceptron model({kHiddenSize, kOutSize}, F::sigmoid_lambda);

  // Optimizer
  const O::StochasticGradientDescent optimizer(kLearningRate);

  for (std::size_t i = 0; i < kIterations; ++i) {
    // Prediction
    const tw::TensorSharedPtr y_pred_ptr = model.Predict({x_ptr})[0];

    // Loss
    const tw::TensorSharedPtr loss_ptr = F::mean_squared_error(y_ptr, y_pred_ptr);

    // Backpropagation
    model.ClearGrads();
    loss_ptr->Backpropagation();

    // Parameter update
    optimizer.Update(model.GetParamPtrs());

    if (i % (kIterations / 10) == 0) {
      DEBUG_PRINT(i);
      DEBUG_PRINT(loss_ptr);
      std::cout << std::endl;
    }
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  for (const auto& param_ptr : model.GetParamPtrs()) {
    DEBUG_PRINT(param_ptr);
    std::cout << std::endl;
  }

  std::cout << "xs = [";
  for (const auto& x : x_ptr->data()) {
    std::cout << x << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "ys = [";
  for (const auto& y : y_ptr->data()) {
    std::cout << y << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  const tw::TensorSharedPtr y_pred_ptr = model.Predict({x_ptr})[0];
  std::cout << "ys_pred = [";
  for (const auto& y_pred : y_pred_ptr->data()) {
    std::cout << y_pred << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

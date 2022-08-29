#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>

#include "tensorward/core.h"
#include "tensorward/dataset.h"
#include "tensorward/function.h"
#include "tensorward/model.h"
#include "tensorward/optimizer.h"
#include "tensorward/util.h"

#define DEBUG_PRINT_SCALAR(var) std::cout << #var << " = " << var << std::endl;
#define DEBUG_PRINT_TENSOR(var) std::cout << #var << " = \n" << var << std::endl;

namespace tw = tensorward::core;
namespace D = tensorward::dataset;
namespace F = tensorward::function;
namespace M = tensorward::model;
namespace O = tensorward::optimizer;
namespace U = tensorward::util;

int main(int argc, char* argv[]) {
  constexpr std::size_t kHiddenSize = 1000;
  constexpr std::size_t kOutSize = 10;

  constexpr std::size_t kMaxEpoch = 5;
  constexpr std::size_t kBatchSize = 100;
  constexpr float kLearningRate = 0.01;
  constexpr float kMomentum = 0.9;

  const tw::TransformLambda flatten_lambda = [](const xt::xarray<float>& input_data) {
    return xt::flatten(input_data);
  };
  const tw::TransformLambda normalize_lambda = [](const xt::xarray<float>& input_data) {
    const float mean = 0.0;
    const float stddev = 255.0;
    return (input_data - mean) / stddev;
  };
  const std::vector<tw::TransformLambda> data_transform_lambdas({flatten_lambda, normalize_lambda});

  // Dataset
  const tw::DatasetSharedPtr train_dataset_ptr =
      tw::AsDatasetSharedPtr<D::Mnist>(/* is_training_mode = */ true, data_transform_lambdas);
  const tw::DatasetSharedPtr test_dataset_ptr =
      tw::AsDatasetSharedPtr<D::Mnist>(/* is_training_mode = */ false, data_transform_lambdas);

  // Decimates the size of the dataset (to 1/10) and gets each batch of the dataset through `DataLoader` class.
  constexpr std::size_t kDecimatingScale = 10;
  tw::DataLoader train_data_loader(train_dataset_ptr, kBatchSize, /* does_shuffle_dataset = */ true, kDecimatingScale);
  tw::DataLoader test_data_loader(test_dataset_ptr, kBatchSize, /* does_shuffle_dataset = */ false, kDecimatingScale);

  DEBUG_PRINT_SCALAR(xt::adapt(train_data_loader.dataset_ptr()->data().shape()));
  DEBUG_PRINT_SCALAR(xt::adapt(train_data_loader.dataset_ptr()->label().shape()));
  DEBUG_PRINT_SCALAR(train_data_loader.dataset_size());
  DEBUG_PRINT_SCALAR(train_data_loader.batch_size());
  DEBUG_PRINT_SCALAR(train_data_loader.max_iteration());
  std::cout << std::endl;

  DEBUG_PRINT_SCALAR(xt::adapt(test_data_loader.dataset_ptr()->data().shape()));
  DEBUG_PRINT_SCALAR(xt::adapt(test_data_loader.dataset_ptr()->label().shape()));
  DEBUG_PRINT_SCALAR(test_data_loader.dataset_size());
  DEBUG_PRINT_SCALAR(test_data_loader.batch_size());
  DEBUG_PRINT_SCALAR(test_data_loader.max_iteration());
  std::cout << std::endl;

  // Model
  M::MultiLayerPerceptron model({kHiddenSize, kHiddenSize, kOutSize}, tw::AsFunctionSharedPtr<F::ReLU>());

  // Optimizer
  O::MomentumStochasticGradientDescent optimizer(kLearningRate, kMomentum);

  std::vector<float> average_train_losses;
  average_train_losses.reserve(kMaxEpoch);
  std::vector<float> average_train_accuracies;
  average_train_accuracies.reserve(kMaxEpoch);

  std::vector<float> average_test_losses;
  average_test_losses.reserve(kMaxEpoch);
  std::vector<float> average_test_accuracies;
  average_test_accuracies.reserve(kMaxEpoch);

  for (std::size_t epoch = 0; epoch < kMaxEpoch; ++epoch) {
    //// Train ////
    std::cout << "Train..." << std::endl;
    float sum_train_loss = 0.0;
    float sum_train_accuracy = 0.0;

    for (std::size_t i = 0; i < train_data_loader.max_iteration(); ++i) {
      // Variables
      const auto [batch_x, batch_t] = train_data_loader.GetBatchAt(i);
      const tw::TensorSharedPtr batch_x_ptr = tw::AsTensorSharedPtr(batch_x, "batch_x");
      const tw::TensorSharedPtr batch_t_ptr = tw::AsTensorSharedPtr(batch_t, "batch_t");

      // Prediction
      const tw::TensorSharedPtr batch_y_pred_ptr = model.Predict({batch_x_ptr})[0];

      // Loss
      const tw::TensorSharedPtr batch_loss_ptr = F::softmax_cross_entropy_error(batch_y_pred_ptr, batch_t_ptr);

      // Backpropagation
      model.ClearGrads();
      batch_loss_ptr->Backpropagation();

      // Parameter update
      optimizer.Update(model.GetParamPtrs());

      const float train_loss = batch_loss_ptr->data()(0) * kBatchSize;
      sum_train_loss += train_loss;

      const float train_accuracy = U::Accuracy(batch_y_pred_ptr->data(), batch_t_ptr->data()) * kBatchSize;
      sum_train_accuracy += train_accuracy;
    }

    const float average_train_loss = sum_train_loss / train_data_loader.dataset_size();
    average_train_losses.push_back(average_train_loss);

    const float average_train_accuracy = sum_train_accuracy / train_data_loader.dataset_size();
    average_train_accuracies.push_back(average_train_accuracy);

    //// Test ////
    std::cout << "Test..." << std::endl;
    float sum_test_loss = 0.0;
    float sum_test_accuracy = 0.0;

    for (std::size_t i = 0; i < test_data_loader.max_iteration(); ++i) {
      {
        tw::UseConfig with(tw::Config::kDoesEnableBackpropagation, false);

        // Variables
        const auto [batch_x, batch_t] = test_data_loader.GetBatchAt(i);
        const tw::TensorSharedPtr batch_x_ptr = tw::AsTensorSharedPtr(batch_x, "batch_x");
        const tw::TensorSharedPtr batch_t_ptr = tw::AsTensorSharedPtr(batch_t, "batch_t");

        // Prediction
        const tw::TensorSharedPtr batch_y_pred_ptr = model.Predict({batch_x_ptr})[0];

        // Loss
        const tw::TensorSharedPtr batch_loss_ptr = F::softmax_cross_entropy_error(batch_y_pred_ptr, batch_t_ptr);

        const float test_loss = batch_loss_ptr->data()(0) * kBatchSize;
        sum_test_loss += test_loss;

        const float test_accuracy = U::Accuracy(batch_y_pred_ptr->data(), batch_t_ptr->data()) * kBatchSize;
        sum_test_accuracy += test_accuracy;
      }
    }

    const float average_test_loss = sum_test_loss / test_data_loader.dataset_size();
    average_test_losses.push_back(average_test_loss);

    const float average_test_accuracy = sum_test_accuracy / test_data_loader.dataset_size();
    average_test_accuracies.push_back(average_test_accuracy);

    //// Print ////
    DEBUG_PRINT_SCALAR(epoch);
    DEBUG_PRINT_SCALAR(average_train_loss);
    DEBUG_PRINT_SCALAR(average_test_loss);
    DEBUG_PRINT_SCALAR(average_train_accuracy);
    DEBUG_PRINT_SCALAR(average_test_accuracy);
    std::cout << std::endl;
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  std::cout << "average_train_losses = [";
  for (const auto& average_train_loss : average_train_losses) {
    std::cout << average_train_loss << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "average_test_losses = [";
  for (const auto& average_test_loss : average_test_losses) {
    std::cout << average_test_loss << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "------------------------------------" << std::endl << std::endl;

  std::cout << "average_train_accuracies = [";
  for (const auto& average_train_loss : average_train_accuracies) {
    std::cout << average_train_loss << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "average_test_accuracies = [";
  for (const auto& average_test_loss : average_test_accuracies) {
    std::cout << average_test_loss << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

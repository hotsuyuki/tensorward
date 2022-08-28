#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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
  constexpr std::size_t kInSize = 2;
  constexpr std::size_t kHiddenSize = 10;
  constexpr std::size_t kOutSize = 3;

  constexpr std::size_t kMaxEpoch = 300;
  constexpr std::size_t kBatchSize = 30;
  constexpr float kLearningRate = 1.0;

  // Dataset
  const tw::DatasetSharedPtr train_dataset = tw::AsDatasetSharedPtr<D::Spiral>(/* is_training_mode = */ true);
  const tw::DatasetSharedPtr test_dataset = tw::AsDatasetSharedPtr<D::Spiral>(/* is_training_mode = */ false);
  tw::DataLoader train_data_loader(train_dataset, kBatchSize, /* does_shuffle_dataset = */ true);
  tw::DataLoader test_data_loader(test_dataset, kBatchSize, /* does_shuffle_dataset = */ false);

  // Model
  M::MultiLayerPerceptron model({kHiddenSize, kOutSize}, tw::AsFunctionSharedPtr<F::Sigmoid>());

  // Optimizer
  O::StochasticGradientDescent optimizer(kLearningRate);

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
    if (epoch % (kMaxEpoch / 10) == 0) {
      DEBUG_PRINT_SCALAR(epoch);
      DEBUG_PRINT_SCALAR(average_train_loss);
      DEBUG_PRINT_SCALAR(average_test_loss);
      DEBUG_PRINT_SCALAR(average_train_accuracy);
      DEBUG_PRINT_SCALAR(average_test_accuracy);
      std::cout << std::endl;
    }
  }

  std::cout << "------------------------------------" << std::endl << std::endl;

  std::cout << "train_dataset_data_xs = [";
  for (std::size_t i = 0; i < train_data_loader.dataset_size(); ++i) {
    const auto [ith_data, _] = train_data_loader.dataset_ptr()->at(i);
    std::cout << ith_data(0) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "train_dataset_data_ys = [";
  for (std::size_t i = 0; i < train_data_loader.dataset_size(); ++i) {
    const auto [ith_data, _] = train_data_loader.dataset_ptr()->at(i);
    std::cout << ith_data(1) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "train_dataset_label_ts = [";
  for (std::size_t i = 0; i < train_data_loader.dataset_size(); ++i) {
    const auto [_, ith_label] = train_data_loader.dataset_ptr()->at(i);
    std::cout << static_cast<std::size_t>(ith_label(0)) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "------------------------------------" << std::endl << std::endl;

  constexpr float kDecisionBoundaryMinX = -1.0;
  constexpr float kDecisionBoundaryMaxX = 1.0;
  constexpr float kDecisionBoundaryDeltaX = 0.1;
  constexpr std::size_t kDecisionBoundaryDataSizeX =
      (kDecisionBoundaryMaxX - kDecisionBoundaryMinX) / kDecisionBoundaryDeltaX;

  constexpr float kDecisionBoundaryMinY = -1.0;
  constexpr float kDecisionBoundaryMaxY = 1.0;
  constexpr float kDecisionBoundaryDeltaY = 0.1;
  constexpr std::size_t kDecisionBoundaryDataSizeY =
      (kDecisionBoundaryMaxY - kDecisionBoundaryMinY) / kDecisionBoundaryDeltaY;

  constexpr std::size_t kDecisionBoundaryDataSize = kDecisionBoundaryDataSizeX * kDecisionBoundaryDataSizeY;

  xt::xarray<float> decision_boundary_data = xt::zeros<float>({kDecisionBoundaryDataSize, kInSize});
  for (std::size_t y = 0; y < kDecisionBoundaryDataSizeY; ++y) {
    for (std::size_t x = 0; x < kDecisionBoundaryDataSizeX; ++x) {
      const std::size_t index = y * kDecisionBoundaryDataSizeX + x;
      xt::view(decision_boundary_data, index) = xt::xarray<float>(
          {kDecisionBoundaryMinX + kDecisionBoundaryDeltaX * x, kDecisionBoundaryMinY + kDecisionBoundaryDeltaY * y});
    }
  }

  std::cout << "decision_boundary_data_xs = [";
  for (std::size_t i = 0; i < decision_boundary_data.shape(0); ++i) {
    std::cout << decision_boundary_data(i, 0) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  std::cout << "decision_boundary_data_ys = [";
  for (std::size_t i = 0; i < decision_boundary_data.shape(0); ++i) {
    std::cout << decision_boundary_data(i, 1) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

  const tw::TensorSharedPtr decision_boundary_data_ptr = tw::AsTensorSharedPtr(decision_boundary_data);
  const tw::TensorSharedPtr decision_boundary_pred_ptr = model.Predict({decision_boundary_data_ptr})[0];
  std::cout << "decision_boundary_pred_ts = [";
  for (std::size_t i = 0; i < decision_boundary_pred_ptr->data().shape(0); ++i) {
    std::cout << xt::argmax(xt::view(decision_boundary_pred_ptr->data(), i)) << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << std::endl;

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

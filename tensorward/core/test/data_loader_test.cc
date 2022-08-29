#include "tensorward/core/data_loader.h"

#include <gtest/gtest.h>
#include <xtensor/xview.hpp>

#include "tensorward/dataset/spiral.h"

namespace tensorward::core {

namespace {

constexpr bool kIsTrainingMode = true;
constexpr std::size_t kBatchSize = 10;
constexpr bool kDoesShuffleDataset = true;
constexpr bool kDoesNotShuffleDataset = false;
constexpr std::size_t kDecimatingScale = 10;

}  // namespace

class DataLoaderTest : public ::testing::Test {
 protected:
  DataLoaderTest()
      : spiral_dataset_ptr_(AsDatasetSharedPtr<dataset::Spiral>(kIsTrainingMode)),
        spiral_data_loader_with_shuffle_(spiral_dataset_ptr_, kBatchSize, kDoesShuffleDataset, kDecimatingScale),
        spiral_data_loader_without_shuffle_(spiral_dataset_ptr_, kBatchSize, kDoesNotShuffleDataset, kDecimatingScale) {
    const std::size_t full_dataset_size = spiral_dataset_ptr_->size();
    const xt::xarray<std::size_t> full_indices_without_shuffle = xt::arange(full_dataset_size);

    const std::size_t decimated_dataset_size = full_dataset_size / kDecimatingScale;
    decimated_indices_without_shuffle_ = xt::zeros<std::size_t>({decimated_dataset_size});

    xt::xarray<float>::shape_type decimated_data_shape = spiral_dataset_ptr_->data().shape();
    decimated_data_shape[0] = decimated_dataset_size;
    decimated_spiral_dataset_data_without_shuffle_ = xt::zeros<std::size_t>(decimated_data_shape);

    xt::xarray<float>::shape_type decimated_label_shape = spiral_dataset_ptr_->label().shape();
    decimated_label_shape[0] = decimated_dataset_size;
    decimated_spiral_dataset_label_without_shuffle_ = xt::zeros<std::size_t>(decimated_label_shape);

    for (std::size_t i = 0; i < decimated_dataset_size; ++i) {
      xt::view(decimated_indices_without_shuffle_, i) = xt::view(full_indices_without_shuffle, kDecimatingScale * i);

      xt::view(decimated_spiral_dataset_data_without_shuffle_, i) =
          xt::view(spiral_dataset_ptr_->data(), kDecimatingScale * i);
      xt::view(decimated_spiral_dataset_label_without_shuffle_, i) =
          xt::view(spiral_dataset_ptr_->label(), kDecimatingScale * i);
    }
  }

  const DatasetSharedPtr spiral_dataset_ptr_;
  DataLoader spiral_data_loader_with_shuffle_;
  DataLoader spiral_data_loader_without_shuffle_;
  xt::xarray<std::size_t> decimated_indices_without_shuffle_;
  xt::xarray<float> decimated_spiral_dataset_data_without_shuffle_;
  xt::xarray<float> decimated_spiral_dataset_label_without_shuffle_;
};

TEST_F(DataLoaderTest, InitTest) {
  // Checks that the indicies in the data loader (with shuffle) are shuffled after `Init()`.
  spiral_data_loader_with_shuffle_.Init();
  EXPECT_NE(spiral_data_loader_with_shuffle_.indices(), decimated_indices_without_shuffle_);

  // Checks that the indicies in the data loader (without shuffle) are not shuffled after `Init()`.
  spiral_data_loader_without_shuffle_.Init();
  EXPECT_EQ(spiral_data_loader_without_shuffle_.indices(), decimated_indices_without_shuffle_);
}

TEST_F(DataLoaderTest, GetBatchAtTest) {
  ASSERT_EQ(spiral_data_loader_with_shuffle_.max_iteration(), spiral_data_loader_without_shuffle_.max_iteration());

  for (std::size_t i = 0; i < spiral_data_loader_with_shuffle_.max_iteration(); ++i) {
    const auto [actual_batch_data_with_shuffle, actual_batch_label_with_shuffle] =
        spiral_data_loader_with_shuffle_.GetBatchAt(i);

    const auto [actual_batch_data_without_shuffle, actual_batch_label_without_shuffle] =
        spiral_data_loader_without_shuffle_.GetBatchAt(i);

    const xt::xarray<float> expected_batch_data_without_shuffle =
        xt::view(decimated_spiral_dataset_data_without_shuffle_, xt::range(i * kBatchSize, (i + 1) * kBatchSize));
    const xt::xarray<float> expected_batch_label_without_shuffle =
        xt::view(decimated_spiral_dataset_label_without_shuffle_, xt::range(i * kBatchSize, (i + 1) * kBatchSize));

    // Checks that the actual batch data and label (with shuffle) are correct at least in the shape level.
    EXPECT_EQ(actual_batch_data_with_shuffle.shape(), expected_batch_data_without_shuffle.shape());
    EXPECT_EQ(actual_batch_label_with_shuffle.shape(), expected_batch_label_without_shuffle.shape());

    // Checks that the actual batch data and label (without shuffle) are correct in the both shape and value level.
    EXPECT_EQ(actual_batch_data_without_shuffle, expected_batch_data_without_shuffle);
    EXPECT_EQ(actual_batch_label_without_shuffle, expected_batch_label_without_shuffle);
  }
}

}  // namespace tensorward::core

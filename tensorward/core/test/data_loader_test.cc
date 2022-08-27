#include "tensorward/core/data_loader.h"

#include <gtest/gtest.h>
#include <xtensor/xview.hpp>

#include "tensorward/dataset/spiral.h"

namespace tensorward::core {

namespace {

constexpr bool kIsTrainingMode = true;
constexpr std::size_t kBatchSize = 10;

}  // namespace

class DatasetTest : public ::testing::Test {
 protected:
  DatasetTest()
      : spiral_dataset_ptr_(AsDatasetSharedPtr<dataset::Spiral>(kIsTrainingMode)),
        spiral_data_loader_with_shuffle_(spiral_dataset_ptr_, kBatchSize, /* does_shuffle_dataset = */ true),
        spiral_data_loader_without_shuffle_(spiral_dataset_ptr_, kBatchSize, /* does_shuffle_dataset = */ false) {}

  const DatasetSharedPtr spiral_dataset_ptr_;
  DataLoader spiral_data_loader_with_shuffle_;
  DataLoader spiral_data_loader_without_shuffle_;
};

TEST_F(DatasetTest, ResetTest) {
  const xt::xarray<std::size_t> non_shuffled_indices = xt::arange(spiral_dataset_ptr_->size());

  // Checks that the indicies in the data loader (with shuffle) are shuffled after `Init()`.
  spiral_data_loader_with_shuffle_.Init();
  EXPECT_NE(spiral_data_loader_with_shuffle_.indices(), non_shuffled_indices);

  // Checks that the indicies in the data loader (without shuffle) are not shuffled after `Init()`.
  spiral_data_loader_without_shuffle_.Init();
  EXPECT_EQ(spiral_data_loader_without_shuffle_.indices(), non_shuffled_indices);
}

TEST_F(DatasetTest, GetBatchAtTest) {
  ASSERT_EQ(spiral_data_loader_with_shuffle_.max_iteration(), spiral_data_loader_without_shuffle_.max_iteration());

  for (std::size_t i = 0; i < spiral_data_loader_with_shuffle_.max_iteration(); ++i) {
    const auto [actual_batch_data_with_shuffle, actual_batch_label_with_shuffle] =
        spiral_data_loader_with_shuffle_.GetBatchAt(i);

    const auto [actual_batch_data_without_shuffle, actual_batch_label_without_shuffle] =
        spiral_data_loader_without_shuffle_.GetBatchAt(i);

    const xt::xarray<float> expected_batch_data_without_shuffle =
        xt::view(spiral_dataset_ptr_->data(), xt::range(i * kBatchSize, (i + 1) * kBatchSize));
    const xt::xarray<float> expected_batch_label_without_shuffle =
        xt::view(spiral_dataset_ptr_->label(), xt::range(i * kBatchSize, (i + 1) * kBatchSize));

    // Checks that the actual batch data and label (with shuffle) are correct at least in the shape level.
    EXPECT_EQ(actual_batch_data_with_shuffle.shape(), expected_batch_data_without_shuffle.shape());
    EXPECT_EQ(actual_batch_label_with_shuffle.shape(), expected_batch_label_without_shuffle.shape());

    // Checks that the actual batch data and label (without shuffle) are correct in the both shape and value level.
    EXPECT_EQ(actual_batch_data_without_shuffle, expected_batch_data_without_shuffle);
    EXPECT_EQ(actual_batch_label_without_shuffle, expected_batch_label_without_shuffle);
  }
}

}  // namespace tensorward::core

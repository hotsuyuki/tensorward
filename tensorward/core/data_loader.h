#pragma once

#include <cmath>
#include <utility>

#include <xtensor/xarray.hpp>

#include "tensorward/core/dataset.h"

namespace tensorward::core {

class DataLoader {
 public:
  DataLoader(const DatasetSharedPtr dataset_ptr, const std::size_t batch_size, const bool does_shuffle_dataset,
             const std::size_t decimating_scale = 1)
      : dataset_ptr_(dataset_ptr),
        batch_size_(batch_size),
        does_shuffle_dataset_(does_shuffle_dataset),
        decimating_scale_(decimating_scale) {
    Init();
  }

  ~DataLoader() {}

  void Init();

  const std::pair<xt::xarray<float>, xt::xarray<float>> GetBatchAt(const std::size_t i);

  const std::size_t dataset_size() const { return dataset_ptr_->size() / decimating_scale_; }

  const std::size_t max_iteration() const {
    return std::ceil(static_cast<float>(dataset_size()) / static_cast<float>(batch_size_));
  }

  const DatasetSharedPtr dataset_ptr() const { return dataset_ptr_; }

  const std::size_t batch_size() const { return batch_size_; }

  const bool does_shuffle_dataset() const { return does_shuffle_dataset_; }

  const std::size_t decimating_scale() const { return decimating_scale_; }

  const xt::xarray<std::size_t>& decimated_indices() const { return decimated_indices_; }

  const xt::xarray<std::size_t>& indices() const { return indices_; }

 private:
  DatasetSharedPtr dataset_ptr_;

  std::size_t batch_size_;

  bool does_shuffle_dataset_;

  std::size_t decimating_scale_;

  xt::xarray<std::size_t> decimated_indices_;

  xt::xarray<std::size_t> indices_;
};

}  // namespace tensorward::core

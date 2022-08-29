#include "tensorward/core/data_loader.h"

#include <cassert>

#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

namespace tensorward::core {

void DataLoader::Init() {
  assert((static_cast<void>("`decimating_scale_` must be a value such that `dataset_ptr_->size()` is divisible by it."),
          dataset_ptr_->size() % decimating_scale_ == 0));

  const std::size_t full_dataset_size = dataset_ptr_->size();
  const xt::xarray<std::size_t> full_indices = xt::arange(full_dataset_size);

  const bool does_decimate_dataset = (2 <= decimating_scale_);
  const bool is_decimated_indices_empty = (decimated_indices_ == xt::xarray<std::size_t>());

  if (does_decimate_dataset && is_decimated_indices_empty) {
    const std::size_t decimated_dataset_size = full_dataset_size / decimating_scale_;
    decimated_indices_ = xt::zeros<std::size_t>({decimated_dataset_size});
    for (std::size_t i = 0; i < decimated_dataset_size; ++i) {
      xt::view(decimated_indices_, i) = xt::view(full_indices, decimating_scale_ * i);
    }
  }

  indices_ = (does_decimate_dataset) ? decimated_indices_ : full_indices;
  if (does_shuffle_dataset_) {
    xt::random::shuffle(indices_);
  }
}

const std::pair<xt::xarray<float>, xt::xarray<float>> DataLoader::GetBatchAt(const std::size_t i) {
  const xt::xarray<std::size_t> batch_indices = xt::view(indices_, xt::range(i * batch_size_, (i + 1) * batch_size_));

  const auto [first_batch_data, first_batch_label] = dataset_ptr_->at(batch_indices(0));

  xt::xarray<float>::shape_type batch_data_shape = first_batch_data.shape();
  batch_data_shape.insert(batch_data_shape.begin(), batch_size_);
  xt::xarray<float> batch_data = xt::zeros<float>(batch_data_shape);

  xt::xarray<float>::shape_type batch_label_shape = first_batch_label.shape();
  batch_label_shape.insert(batch_label_shape.begin(), batch_size_);
  xt::xarray<float> batch_label = xt::zeros<float>(batch_label_shape);

  for (std::size_t i_batch_indices = 0; i_batch_indices < batch_indices.size(); ++i_batch_indices) {
    const auto [ith_batch_data, ith_batch_label] = dataset_ptr_->at(batch_indices(i_batch_indices));
    xt::view(batch_data, i_batch_indices) = ith_batch_data;
    xt::view(batch_label, i_batch_indices) = ith_batch_label;
  }

  assert(batch_data.shape(0) == batch_size_);
  assert(batch_label.shape(0) == batch_size_);

  const std::pair<xt::xarray<float>, xt::xarray<float>> batch_data_label_pair(batch_data, batch_label);

  if (i == (max_iteration() - 1)) {
    Init();
  }

  return batch_data_label_pair;
}

}  // namespace tensorward::core

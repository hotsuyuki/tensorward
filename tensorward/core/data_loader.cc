#include "tensorward/core/data_loader.h"

#include <cassert>

#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

namespace tensorward::core {

void DataLoader::Init() {
  indices_ = does_shuffle_dataset_ ? xt::random::permutation(dataset_ptr_->size()) : xt::arange(dataset_ptr_->size());
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

  if (i == (max_iteration_ - 1)) {
    Init();
  }

  return batch_data_label_pair;
}

}  // namespace tensorward::core

#include "tensorward/core/data_loader.h"

#include <cassert>

#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

namespace tensorward::core {

void DataLoader::Reset() {
  indices_ = does_shuffle_dataset_ ? xt::random::permutation(dataset_ptr_->size()) : xt::arange(dataset_ptr_->size());
}

const std::pair<xt::xarray<float>, xt::xarray<float>> DataLoader::GetBatchAt(const std::size_t i) {
  const xt::xarray<std::size_t> batch_indices = xt::view(indices_, xt::range(i * batch_size_, (i + 1) * batch_size_));

  const auto [first_data, first_label] = dataset_ptr_->at(batch_indices(0));
  xt::xarray<float> batch_data = xt::expand_dims(first_data, 0);
  xt::xarray<float> batch_label = xt::expand_dims(first_label, 0);

  for (std::size_t i_batch_indices = 1; i_batch_indices < batch_indices.size(); ++i_batch_indices) {
    const auto [one_data, one_label] = dataset_ptr_->at(batch_indices(i_batch_indices));
    batch_data = xt::concatenate(xt::xtuple(batch_data, xt::expand_dims(one_data, 0)));
    batch_label = xt::concatenate(xt::xtuple(batch_label, xt::expand_dims(one_label, 0)));
  }

  assert(batch_data.dimension() == dataset_ptr_->data().dimension());
  assert(batch_data.shape(0) == batch_size_);

  assert(batch_label.dimension() == dataset_ptr_->label().dimension());
  assert(batch_label.shape(0) == batch_size_);

  const std::pair<xt::xarray<float>, xt::xarray<float>> batch_data_label_pair(batch_data, batch_label);

  if (i == (max_iteration_ - 1)) {
    Reset();
  }

  return batch_data_label_pair;
}

}  // namespace tensorward::core

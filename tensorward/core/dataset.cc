#include "tensorward/core/dataset.h"

#include <cassert>

#include <xtensor/xview.hpp>

namespace tensorward::core {

const std::size_t Dataset::size() const {
  assert((static_cast<void>("The size of `data_` and `label_` must be equal."), data_.shape(0) == label_.shape(0)));

  return data_.shape(0);
}

const std::pair<xt::xarray<float>, xt::xarray<float>> Dataset::at(const std::size_t i) const {
  const xt::xarray<float>& ith_data = xt::view(data_, i);
  const xt::xarray<float>& ith_label = xt::view(label_, i);
  const std::pair<xt::xarray<float>, xt::xarray<float>> transformed_ith_data_label_pair =
      ApplyTransformLambdas(ith_data, ith_label);

  return transformed_ith_data_label_pair;
}

const std::pair<xt::xarray<float>, xt::xarray<float>> Dataset::ApplyTransformLambdas(
    const xt::xarray<float>& ith_data, const xt::xarray<float>& ith_label) const {
  // Uses the copy construct in order to avoid modifying the original data and label when applying transform lambdas.
  xt::xarray<float> transformed_ith_data(ith_data);
  xt::xarray<float> transformed_ith_label(ith_label);

  for (const auto& data_transform_lambda : data_transform_lambdas_) {
    transformed_ith_data = data_transform_lambda(transformed_ith_data);
  }
  for (const auto& label_transform_lambda : label_transform_lambdas_) {
    transformed_ith_label = label_transform_lambda(transformed_ith_label);
  }

  const std::pair<xt::xarray<float>, xt::xarray<float>> transformed_ith_data_label_pair(transformed_ith_data,
                                                                                        transformed_ith_label);

  return transformed_ith_data_label_pair;
}

}  // namespace tensorward::core

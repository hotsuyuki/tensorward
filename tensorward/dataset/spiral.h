#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "tensorward/core/dataset.h"

namespace tensorward::dataset {

class Spiral : public core::Dataset {
 public:
  Spiral(const bool is_training_mode, const std::vector<core::TransformLambda>& data_transform_lambdas = {},
         const std::vector<core::TransformLambda>& label_transform_lambdas = {})
      : core::Dataset(is_training_mode, data_transform_lambdas, label_transform_lambdas),
        in_size_(2),
        class_size_(3),
        data_size_for_each_class_(100),
        data_size_(class_size_ * data_size_for_each_class_) {
    Prepare();
  }

  ~Spiral() {}

  void Prepare() override {
    const int seed = is_training_mode_ ? 1984 : 2020;
    xt::random::seed(seed);

    xt::xarray<float> data = xt::zeros<float>({data_size_, in_size_});
    xt::xarray<float> label = xt::zeros<float>({data_size_});  // non-onehot label

    // Populates the data and label.
    for (std::size_t i_class = 0; i_class < class_size_; ++i_class) {
      for (std::size_t i_data = 0; i_data < data_size_for_each_class_; ++i_data) {
        const float radius = static_cast<float>(i_data) / static_cast<float>(data_size_for_each_class_);

        const float noise = 0.2 * xt::random::randn<float>({1})(0);
        const float theta = 4.0 * i_class + 4.0 * radius + noise;

        const std::size_t i = i_class * data_size_for_each_class_ + i_data;

        xt::view(data, i) = xt::xarray<float>({radius * std::cos(theta), radius * std::sin(theta)});
        xt::view(label, i) = i_class;
      }
    }

    data_ = xt::zeros<float>({data_size_, in_size_});
    label_ = xt::zeros<float>({data_size_});  // non-onehot label

    // Shuffles the data and label.
    const xt::xarray<std::size_t> shuffled_index = xt::random::permutation(data_size_);
    for (std::size_t i = 0; i < data_size_; ++i) {
      xt::view(data_, i) = xt::view(data, shuffled_index(i));
      xt::view(label_, i) = xt::view(label, shuffled_index(i));
    }
  }

  const std::size_t in_size() const { return in_size_; }

  const std::size_t class_size() const { return class_size_; }

  const std::size_t data_size_for_each_class() const { return data_size_for_each_class_; }

  const std::size_t data_size() const { return data_size_; }

 private:
  std::size_t in_size_;

  std::size_t class_size_;

  std::size_t data_size_for_each_class_;

  std::size_t data_size_;
};

}  // namespace tensorward::dataset

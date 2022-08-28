#pragma once

#include <cassert>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "tensorward/dataset/mnist.h"

namespace tensorward::dataset {

class MnistOneTenth : public Mnist {
 public:
  MnistOneTenth(const bool is_training_mode, const std::vector<core::TransformLambda>& data_transform_lambdas,
        const std::vector<core::TransformLambda>& label_transform_lambdas)
      : Mnist(is_training_mode, data_transform_lambdas, label_transform_lambdas), reducing_scale_(10) {
    // N' = N / rs
    assert(data_.shape(0) == label_.shape(0));
    assert(data_.shape(0) % reducing_scale_ == 0);
    const std::size_t reduced_dataset_size = data_.shape(0) / reducing_scale_;

    // {N, C, H, W} ---> {N', C, H, W}
    xt::xarray<float>::shape_type reduced_data_shape = data_.shape();
    reduced_data_shape[0] = reduced_dataset_size;
    xt::xarray<float> reduced_data = xt::zeros<float>(reduced_data_shape);

    // {N} ---> {N'}
    xt::xarray<float>::shape_type reduced_label_shape = label_.shape();
    reduced_label_shape[0] = reduced_dataset_size;
    xt::xarray<float> reduced_label = xt::zeros<float>(reduced_label_shape);

    // Decimates the original data and label.
    for (std::size_t i = 0; i < reduced_dataset_size; ++i) {
      xt::view(reduced_data, i) = xt::view(data_, reducing_scale_ * i);
      xt::view(reduced_label, i) = xt::view(label_, reducing_scale_ * i);
    }

    // Overwrites the original data and label by the reduced ones.
    data_ = reduced_data;
    label_ = reduced_label;
  }

  ~MnistOneTenth() {}

  const std::size_t reducing_scale() const { return reducing_scale_; }

 private:
  std::size_t reducing_scale_;
};

}  // namespace tensorward::dataset

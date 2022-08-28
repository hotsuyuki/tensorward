#pragma once

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <utility>
#include <vector>

#include <xtensor/xarray.hpp>

namespace tensorward::core {

// TODO: Add `Compose` class that composes transform lambdas in `tensorward::transforms` namespace.
using TransformLambda = std::function<xt::xarray<float>(const xt::xarray<float>&)>;

class Dataset {
 public:
  Dataset(const bool is_training_mode, const std::vector<TransformLambda>& data_transform_lambdas,
          const std::vector<TransformLambda>& label_transform_lambdas,
          const std::filesystem::path& dataset_directory_name)
      : is_training_mode_(is_training_mode),
        data_transform_lambdas_(data_transform_lambdas),
        label_transform_lambdas_(label_transform_lambdas) {
    // Creates a directory for saving the dataset files under "~/.tensorward/dataset/" if it doesn't exist yet.
    dataset_directory_path_ =
        static_cast<std::filesystem::path>(std::getenv("HOME")) / ".tensorward" / "dataset" / dataset_directory_name;
    std::filesystem::create_directories(dataset_directory_path_);
  }

  virtual ~Dataset() {}

  // This member function is pure virtual, so we need to override this in the derived class.
  virtual void Init() = 0;

  // This member function is non-pure virtual, and this has an implementation for the most basic case where
  // `data_` and `label_` store the entire dataset, which means the dataset is small enough to store.
  // We can override this in the derived class as needed (e.g. for a case where the dataset is too large to store).
  virtual const std::size_t size() const;

  // This member function is non-pure virtual, and this has an implementation for the most basic case where
  // `data_` and `label_` store the entire dataset, which means the dataset is small enough to store.
  // We can override this in the derived class as needed (e.g. for a case where the dataset is too large to store).
  virtual const std::pair<xt::xarray<float>, xt::xarray<float>> at(const std::size_t i) const;

  const bool is_training_mode() const { return is_training_mode_; }

  const std::vector<TransformLambda>& data_transform_lambdas() const { return data_transform_lambdas_; }

  const std::vector<TransformLambda>& label_transform_lambdas() const { return label_transform_lambdas_; }

  const std::filesystem::path& dataset_directory_path() const { return dataset_directory_path_; }

  const xt::xarray<float>& data() const { return data_; }

  const xt::xarray<float>& label() const { return label_; }

 protected:
  const std::pair<xt::xarray<float>, xt::xarray<float>> ApplyTransformLambdas(const xt::xarray<float>& ith_data,
                                                                              const xt::xarray<float>& ith_label) const;

  bool is_training_mode_;

  std::vector<TransformLambda> data_transform_lambdas_;

  std::vector<TransformLambda> label_transform_lambdas_;

  std::filesystem::path dataset_directory_path_;

  xt::xarray<float> data_;

  xt::xarray<float> label_;
};

using DatasetSharedPtr = std::shared_ptr<Dataset>;

// NOTE: Because this function is templated, the function definition should be in the header file.
template <class T>
const DatasetSharedPtr AsDatasetSharedPtr(const bool is_training_mode,
                                          const std::vector<TransformLambda>& data_transform_lambdas = {},
                                          const std::vector<TransformLambda>& label_transform_lambdas = {}) {
  return std::make_shared<T>(is_training_mode, data_transform_lambdas, label_transform_lambdas);
}

}  // namespace tensorward::core

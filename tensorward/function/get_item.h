#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"

namespace tensorward::function {

class GetItem : public core::Function {
 public:
  GetItem(const std::vector<xt::xindex>& indices)
      : core::Function({.num_inputs = 1, .num_outputs = 1}), indices_(indices) {}

  ~GetItem() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    const xt::xarray<float> y = xt::index_view(xs[0], indices_);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& x = input_tensor_ptrs_[0]->data();

    xt::xarray<float> dL_dx = xt::zeros_like(x);
    for (std::size_t i = 0; i < indices_.size(); ++i) {
      const std::vector<xt::xindex> ith_index_vector({indices_[i]});
      xt::index_view(dL_dx, ith_index_vector) += dL_dy[i];
    }

    return {dL_dx};
  }

  const std::vector<xt::xindex>& indices() const { return indices_; }

 private:
  std::vector<xt::xindex> indices_;
};

const core::TensorSharedPtr get_item(const core::TensorSharedPtr input_tensor_ptr,
                                     const std::vector<xt::xindex>& indices) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr transpose_function_ptr = std::make_shared<GetItem>(indices);
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs = transpose_function_ptr->Call({input_tensor_ptr});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

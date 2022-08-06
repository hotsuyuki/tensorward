#pragma once

#include <memory>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>

#include "tensorward/core/function.h"
#include "tensorward/core/tensor.h"
#include "tensorward/util/xtensor_cross_entropy_error.h"
#include "tensorward/util/xtensor_softmax.h"

namespace tensorward::function {

class SoftmaxCrossEntropyError : public core::Function {
 public:
  SoftmaxCrossEntropyError() : core::Function({.num_inputs = 2, .num_outputs = 1}) {}

  ~SoftmaxCrossEntropyError() {}

  const std::vector<xt::xarray<float>> Forward(const std::vector<xt::xarray<float>>& xs) override {
    const xt::xarray<float>& x = xs[0];  // score
    const xt::xarray<float>& t = xs[1];  // label

    // p = softmax(x)
    probability_ = xt::clip(util::XtensorSoftmax(x), 1.0e-12, 1.0);  // probability
    const xt::xarray<float>& p = probability_;

    // y = cross_entropy_error(p, t)
    const xt::xarray<float> y = util::XtensorCrossEntropyError(p, t);

    return {y};
  }

  const std::vector<xt::xarray<float>> Backward(const std::vector<xt::xarray<float>>& dL_dys) override {
    const xt::xarray<float>& dL_dy = dL_dys[0];
    const xt::xarray<float>& p = probability_;                   // probability
    const xt::xarray<float>& t = input_tensor_ptrs_[1]->data();  // label

    // TODO: Move this code block to something like `util::GetNumDatasAndNumClasses()`.
    const bool is_multiple_datas = (p.dimension() == 2);
    const std::size_t num_data = is_multiple_datas ? p.shape(0) : 1;

    // y = cross_entropy_error(softmax(x), t) = cross_entropy_error(p, t) ---> dy_dx = (p - t) / N
    xt::xarray<float> dy_dx;
    const bool is_t_onehot = (p.shape() == t.shape());
    if (is_t_onehot) {
      dy_dx = (p - t) / num_data;
    } else {
      // Converts a non-onehot label to indices to extract the corresponding probability.
      //
      // t = {    --->    t_indices = {
      //   0,               { 0,   0 },
      //   1,               { 1,   1 },
      //   2,               { 2,   2 },
      //   :                :
      //   :                :
      //   0                { N-1, 0 }
      // }
      //
      std::vector<xt::xindex> t_indices;
      for (std::size_t i = 0; i < num_data; ++i) {
        t_indices.push_back({i, static_cast<std::size_t>(t(i))});
      }
      dy_dx = p;
      xt::index_view(dy_dx, t_indices) -= 1.0;
      dy_dx = dy_dx / num_data;
    }

    const xt::xarray<float> dL_dx = dL_dy * dy_dx;

    const xt::xarray<float> dL_dt = xt::zeros_like(t);  // Dummy gradient.

    return {dL_dx, dL_dt};
  }

  const xt::xarray<float>& probability() const { return probability_; }

 private:
  xt::xarray<float> probability_;
};

const core::TensorSharedPtr softmax_cross_entropy_error(const core::TensorSharedPtr input_tensor_ptr0,
                                                        const core::TensorSharedPtr input_tensor_ptr1) {
  // Creates an function (dynamically in heap memory so that it's accessible even after exiting this scope), and
  // performs the forward calculation and the computational graph growth.
  const core::FunctionSharedPtr softmax_cross_entropy_error_function_ptr = std::make_shared<SoftmaxCrossEntropyError>();
  const std::vector<core::TensorSharedPtr> output_tensor_ptrs =
      softmax_cross_entropy_error_function_ptr->Call({input_tensor_ptr0, input_tensor_ptr1});

  return output_tensor_ptrs[0];
}

}  // namespace tensorward::function

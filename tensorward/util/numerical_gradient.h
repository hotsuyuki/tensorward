#pragma once

#include <cassert>
#include <vector>

#include <xtensor/xarray.hpp>

#include "tensorward/core/function.h"

namespace tensorward::util {

const std::vector<xt::xarray<float>> NumericalGradient(const core::FunctionSharedPtr function_ptr,
                                                       const std::vector<xt::xarray<float>>& input_datas,
                                                       const float epsilon = 1.0e-3) {
  std::vector<xt::xarray<float>> numerical_grads;
  numerical_grads.reserve(input_datas.size());
  for (std::size_t i = 0; i < input_datas.size(); ++i) {
    std::vector<xt::xarray<float>> positive_input_datas(input_datas);
    positive_input_datas[i] += epsilon;

    std::vector<xt::xarray<float>> negative_input_datas(input_datas);
    negative_input_datas[i] -= epsilon;

    const std::vector<xt::xarray<float>> positive_output_datas = function_ptr->Forward(positive_input_datas);
    const std::vector<xt::xarray<float>> negative_output_datas = function_ptr->Forward(negative_input_datas);

    // TODO: Extend to be capable of handling multiple-outputs-func (Currently it only supports single-output-func)
    assert(positive_output_datas.size() == 1);
    assert(negative_output_datas.size() == 1);

    const xt::xarray<float> numerical_grad = (positive_output_datas[0] - negative_output_datas[0]) / (2.0 * epsilon);
    numerical_grads.push_back(numerical_grad);
  }

  assert(input_datas.size() == numerical_grads.size());
  return numerical_grads;
}

}  // namespace tensorward::util

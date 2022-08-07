#pragma once

#include <cassert>

#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>

namespace tensorward::util {

const float Accuracy(const xt::xarray<float>& score, const xt::xarray<float>& label) {
  const xt::xarray<float> score_argmax = xt::argmax(score, -1);
  assert(score_argmax.dimension() == 1);
  assert(score_argmax.shape(0) == score.shape(0));

  const bool is_onehot_label = (score.shape() == label.shape());
  const xt::xarray<float> label_argmax = is_onehot_label ? xt::xarray<float>(xt::argmax(label, -1)) : label;
  assert(label_argmax.dimension() == 1);
  assert(label_argmax.shape(0) == label.shape(0));

  assert(score_argmax.shape() == label_argmax.shape());
  const xt::xarray<float> equality = xt::equal(score_argmax, label_argmax);
  const float accuracy = xt::mean(equality)(0);

  return accuracy;
}

}  // namespace tensorward::util

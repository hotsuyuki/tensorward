#pragma once

#include <cassert>

#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>

namespace tensorward::util {

const xt::xarray<float> XtensorCrossEntropyError(const xt::xarray<float>& input_data0,
                                                 const xt::xarray<float>& input_data1) {
  // Clips the bottom value with a very small value in order to avoid `log(0.0)`, which is `-infinity`.
  const xt::xarray<float> modified_input_data0 = xt::maximum(input_data0, 1.0e-12);

  // N ... Number of datas
  // C ... Number of classes

  // p ... probability
  //
  // N >= 2:
  // p = {
  //   { p_0_0,     p_0_1,     p_0_2,     ..., p_0_(C-1)     },
  //   { p_1_0,     p_1_1,     p_1_2,     ..., p_0_(C-1)     },
  //   { p_2_0,     p_2_1,     p_2_2,     ..., p_0_(C-1)     },
  //   :
  //   :
  //   { p_(N-1)_0, p_(N-1)_1, p_(N-1)_2, ..., p_(N-1)_(C-1) }
  // }
  //
  // or
  //
  // N == 1:
  // p = { p_0_0, p_0_1, p_0_2, ..., p_0_(C-1) }
  //
  const xt::xarray<float>& p = modified_input_data0;

  // t ... label
  //
  // N >= 2:            N >= 2 (onehot):
  // t = {              t = {
  //   0,                 { 1, 0, 0, ..., 0 },
  //   1,                 { 0, 1, 0, ..., 0 },
  //   2,                 { 0, 0, 1, ..., 0 },
  //   :                  :
  //   :                  :
  //   0                  { 1, 0, 0, ..., 0 }
  // }                  }
  //
  // or
  //
  // N == 1:            N == 1 (onehot):
  // t = 0              t = { 1, 0, 0, ..., 0 }
  //
  const xt::xarray<float>& t = input_data1;

  assert((static_cast<void>("`p.dimension()` must be 2 (multiple datas) or 1 (single data)."),
         (p.dimension() == 2 || p.dimension() == 1)));

  // TODO: Move this code block to something like `util::GetNumDatasAndNumClasses()`.
  const bool is_multiple_datas = (p.dimension() == 2);
  const std::size_t num_data = is_multiple_datas ? p.shape(0) : 1;

  if (is_multiple_datas) {
    assert((static_cast<void>("`t.dimension()` must be 1 (non-onehot) or 2 (onehot) if there are multiple datas."),
           (t.dimension() == 1 || t.dimension() == 2)));
  } else {
    assert((static_cast<void>("`t.dimension()` must be 0 (non-onehot) or 1 (onehot) if there is single data."),
           (t.dimension() == 0 || t.dimension() == 1)));
  }

  xt::xarray<float> t_log_p;
  const bool is_onehot_label = (p.shape() == t.shape());
  if (is_onehot_label) {
    t_log_p = t * xt::log(p);
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
      const xt::xindex t_index({i, static_cast<std::size_t>(t(i))});
      t_indices.push_back(t_index);
    }
    t_log_p = xt::log(xt::index_view(p, t_indices));
  }

  // y = -sum(t * log(p)) / N
  const xt::xarray<float> output_data = -xt::sum(t_log_p) / num_data;

  return output_data;
}

}  // namespace tensorward::util

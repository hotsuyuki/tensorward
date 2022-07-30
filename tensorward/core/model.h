#pragma once

#include <memory>
#include <vector>

#include "tensorward/core/layer.h"
#include "tensorward/core/parameter.h"
#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Model {
 public:
  Model() {}

  virtual ~Model() {}

  // Performs the forward calculation of the function of the layers of this model.
  virtual const std::vector<TensorSharedPtr> Predict(const std::vector<TensorSharedPtr>& input_tensor_ptrs) const = 0;

  void ClearGrads();

  const std::vector<ParameterSharedPtr>& GetParamPtrs();

  // TODO: Implement `Plot()` that plots the computational graph of this model using Graphviz DOT language.

  const std::vector<LayerSharedPtr>& layer_ptrs() const { return layer_ptrs_; }

  const std::vector<ParameterSharedPtr>& param_ptrs() const { return param_ptrs_; }

 protected:
  std::vector<LayerSharedPtr> layer_ptrs_;

  std::vector<ParameterSharedPtr> param_ptrs_;
};

}  // namespace tensorward::core

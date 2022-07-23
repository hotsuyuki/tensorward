#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

#include "tensorward/core/tensor.h"

namespace tensorward::core {

class Parameter : public Tensor {
 public:
  Parameter(const xt::xarray<float>& data, const std::string& name = "") : Tensor(data, name) {}

  ~Parameter() {}
};

using ParameterSharedPtr = std::shared_ptr<Parameter>;

const ParameterSharedPtr AsParameterSharedPtr(const xt::xarray<float>& data, const std::string& name = "");

inline std::ostream& operator<<(std::ostream& os, const Parameter& parameter) {
  os << static_cast<Tensor>(parameter);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const ParameterSharedPtr parameter_ptr) {
  os << static_cast<TensorSharedPtr>(parameter_ptr);
  return os;
}

}  // namespace tensorward::core

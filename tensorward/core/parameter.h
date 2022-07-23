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
  !parameter.name().empty() ? (os << std::endl << "Parameter '" << parameter.name() << "'")
                            : (os << std::endl << "Parameter");
  os << std::endl << "data:" << std::endl << parameter.data();
  if (parameter.grad_opt().has_value()) {
    os << std::endl << "grad:" << std::endl << parameter.grad();
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const ParameterSharedPtr parameter_ptr) {
  parameter_ptr ? (os << *parameter_ptr) : (os << std::endl << "Parameter (Null)");
  return os;
}

}  // namespace tensorward::core

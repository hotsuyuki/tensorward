#include "tensorward/core/parameter.h"

namespace tensorward::core {

const ParameterSharedPtr AsParameterSharedPtr(const xt::xarray<float>& data, const std::string& name /* = "" */) {
  return std::make_shared<Parameter>(data, name);
}

}  // namespace tensorward::core

#pragma once

#include <memory>

namespace tensorward::core {

// Forward declaration in order to avoid circular dependency between `Tensor` class and `Function` class.
class Function;

using FunctionSharedPtr = std::shared_ptr<Function>;

}  // namespace tensorward::core

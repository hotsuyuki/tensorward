#pragma once

#include <memory>

namespace tensorward::core {

// Forward declaration in order to avoid circular dependency between `Tensor` class and `Function` class.
class Tensor;

using TensorSharedPtr = std::shared_ptr<Tensor>;
using TensorWeakPtr = std::weak_ptr<Tensor>;

}  // namespace tensorward::core

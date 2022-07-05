#include "tensorward/core/config.h"

#include <gtest/gtest.h>
#include <xtensor/xrandom.hpp>

#include "tensorward/core/tensor.h"
#include "tensorward/function/exp.h"
#include "tensorward/function/square.h"

namespace tensorward::core {

namespace {

constexpr int kHeight = 2;
constexpr int kWidth = 3;

}  // namespace

class ConfigTest : public ::testing::Test {
 protected:
  ConfigTest()
      : input_tensor_ptr_(AsTensorSharedPtr(xt::random::rand<float>({kHeight, kWidth}))) {}

  const TensorSharedPtr input_tensor_ptr_;
};

TEST_F(ConfigTest, UseConfigTest) {
  constexpr bool kExpectedValueOutsideScope = true;
  constexpr bool kExpectedValueInsideScope = false;

  // Checks that the config value is the default value before entering the scope.
  EXPECT_EQ(Config::instance().config_value(Config::kDoesEnableBackpropagation), kExpectedValueOutsideScope);

  {
    UseConfig with(Config::kDoesEnableBackpropagation, false);

    // Checks that the config value becomes the value set by the UseConfig instance above.
    EXPECT_EQ(Config::instance().config_value(Config::kDoesEnableBackpropagation), kExpectedValueInsideScope);

    const TensorSharedPtr output_tensor_ptr = function::exp(function::square(input_tensor_ptr_));

    // Checks that the computational graph is not created (because the backpropagation is not enabled).
    EXPECT_EQ(output_tensor_ptr->parent_function_ptr(), nullptr);
  }

  // Checks that the config value turns back to the default value after exiting the scope.
  EXPECT_EQ(Config::instance().config_value(Config::kDoesEnableBackpropagation), kExpectedValueOutsideScope);
}

}  // namespace tensorward::core

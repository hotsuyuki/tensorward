load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "config",
  hdrs = ["config.h"],
  visibility = ["//visibility:public"],
)

cc_library(
  # Forward declaration in order to avoid circular dependency between `Tensor` class and `Function` class.
  name = "function_fwd",
  hdrs = ["function_fwd.h"],
  visibility = ["//visibility:private"],
)

cc_library(
  name = "function",
  srcs = ["function.cc"],
  hdrs = [
    "function.h",
    "tensor.h"  # In order to depend on `Tensor` class in `Function` class.
  ],
  deps = [
    ":config",
    ":function_fwd",
    ":tensor_fwd",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  # Forward declaration in order to avoid circular dependency between `Tensor` class and `Function` class.
  name = "tensor_fwd",
  hdrs = ["tensor_fwd.h"],
  visibility = ["//visibility:private"],
)

cc_library(
  name = "tensor",
  srcs = ["tensor.cc"],
  hdrs = [
    "tensor.h",
    "function.h"  # In order to depend on `Function` class in `Tensor` class.
  ],
  deps = [
    ":function_fwd",
    ":tensor_fwd",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "config_test",
  srcs = ["test/config_test.cc"],
  deps = [
    ":config",
    "//tensorward/core:tensor",
    "//tensorward/function:exp",
    "//tensorward/function:square",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "function_test",
  srcs = ["test/function_test.cc"],
  deps = [
    ":function",
    "//tensorward/core:tensor",
    "//tensorward/core/operator:add",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "tensor_test",
  srcs = ["test/tensor_test.cc"],
  deps = [
    ":tensor",
    "//tensorward/core:function",
    "//tensorward/function:exp",
    "//tensorward/function:square",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

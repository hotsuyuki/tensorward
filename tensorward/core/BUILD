load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "config",
  hdrs = ["config.h"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "dataset",
  srcs = ["dataset.cc"],
  hdrs = [
    "dataset.h",
  ],
  deps = [
    "@xtensor//:xtensor",
  ],
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
  name = "layer",
  srcs = ["layer.cc"],
  hdrs = [
    "layer.h",
  ],
  deps = [
    ":parameter",
    ":tensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "model",
  srcs = ["model.cc"],
  hdrs = [
    "model.h",
  ],
  deps = [
    ":layer",
    ":parameter",
    ":tensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "optimizer",
  srcs = ["optimizer.cc"],
  hdrs = [
    "optimizer.h",
  ],
  deps = [
    ":function",
    ":parameter",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "parameter",
  srcs = ["parameter.cc"],
  hdrs = [
    "parameter.h",
  ],
  deps = [
    ":tensor",
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
  name = "dataset_test",
  srcs = ["test/dataset_test.cc"],
  deps = [
    ":dataset",
    "//tensorward/dataset:spiral",
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
  name = "layer_test",
  srcs = ["test/layer_test.cc"],
  deps = [
    ":layer",
    "//tensorward/layer:linear",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "model_test",
  srcs = ["test/model_test.cc"],
  deps = [
    ":model",
    "//tensorward/function:sigmoid",
    "//tensorward/model:multi_layer_perceptron",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "optimizer_test",
  srcs = ["test/optimizer_test.cc"],
  deps = [
    ":optimizer",
    "//tensorward/optimizer:stochastic_gradient_descent",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "parameter_test",
  srcs = ["test/parameter_test.cc"],
  deps = [
    ":parameter",
    "//tensorward/core:tensor",
    "//tensorward/core/operator:add",
    "//tensorward/core/operator:div",
    "//tensorward/core/operator:mul",
    "//tensorward/core/operator:sub",
    "//tensorward/function:mean_squared_error",
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

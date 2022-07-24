load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "multi_layer_perceptron",
  hdrs = ["multi_layer_perceptron.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:layer",
    "//tensorward/core:model",
    "//tensorward/core:tensor",
    "//tensorward/layer:linear",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "multi_layer_perceptron_test",
  srcs = ["test/multi_layer_perceptron_test.cc"],
  deps = [
    ":multi_layer_perceptron",
    "//tensorward/function:sigmoid",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
    "@xtensor-blas//:xtensor-blas",
  ],
)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "momentum_stochastic_gradient_descent",
  hdrs = ["momentum_stochastic_gradient_descent.h"],
  deps = [
    "//tensorward/core:optimizer",
    "//tensorward/core:parameter",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "stochastic_gradient_descent",
  hdrs = ["stochastic_gradient_descent.h"],
  deps = [
    "//tensorward/core:optimizer",
    "//tensorward/core:parameter",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "momentum_stochastic_gradient_descent_test",
  srcs = ["test/momentum_stochastic_gradient_descent_test.cc"],
  deps = [
    ":momentum_stochastic_gradient_descent",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "stochastic_gradient_descent_test",
  srcs = ["test/stochastic_gradient_descent_test.cc"],
  deps = [
    ":stochastic_gradient_descent",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

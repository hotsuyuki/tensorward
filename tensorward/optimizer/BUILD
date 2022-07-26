load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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
  name = "stochastic_gradient_descent_test",
  srcs = ["test/stochastic_gradient_descent_test.cc"],
  deps = [
    ":stochastic_gradient_descent",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

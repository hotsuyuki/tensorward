load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "linear",
  hdrs = ["linear.h"],
  deps = [
    "//tensorward/core:layer",
    "//tensorward/core:parameter",
    "//tensorward/core:tensor",
    "//tensorward/function:linear",
    "//tensorward/function:matmul",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "linear_test",
  srcs = ["test/linear_test.cc"],
  deps = [
    ":linear",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
    "@xtensor-blas//:xtensor-blas",
  ],
)

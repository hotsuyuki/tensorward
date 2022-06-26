load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "numerical_gradient",
  hdrs = ["numerical_gradient.h"],
  deps = [
    "//tensorward/core:function",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "numerical_gradient_test",
  srcs = ["test/numerical_gradient_test.cc"],
  deps = [
    ":numerical_gradient",
    "//tensorward/function:exp",
    "//tensorward/function:pow",
    "//tensorward/function:square",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "exp",
  hdrs = ["exp.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "pow",
  hdrs = ["pow.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "square",
  hdrs = ["square.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "exp_test",
  srcs = ["test/exp_test.cc"],
  deps = [
    ":exp",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "pow_test",
  srcs = ["test/pow_test.cc"],
  deps = [
    ":pow",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "square_test",
  srcs = ["test/square_test.cc"],
  deps = [
    ":square",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

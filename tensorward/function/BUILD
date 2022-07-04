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
  name = "reshape",
  hdrs = ["reshape.h"],
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

cc_library(
  name = "sum",
  hdrs = ["sum.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "transpose",
  hdrs = ["transpose.h"],
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
  name = "reshape_test",
  srcs = ["test/reshape_test.cc"],
  deps = [
    ":reshape",
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

cc_test(
  name = "sum_test",
  srcs = ["test/sum_test.cc"],
  deps = [
    ":sum",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "transpose_test",
  srcs = ["test/transpose_test.cc"],
  deps = [
    ":transpose",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "broadcast_to",
  hdrs = ["broadcast_to.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "//tensorward/util:xtensor_sum_to",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

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
  name = "sum_to",
  hdrs = ["sum_to.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "//tensorward/util:xtensor_sum_to",
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
  name = "broadcast_to_test",
  srcs = ["test/broadcast_to_test.cc"],
  deps = [
    ":broadcast_to",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
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
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "sum_to_test",
  srcs = ["test/sum_to_test.cc"],
  deps = [
    ":sum_to",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "transpose_test",
  srcs = ["test/transpose_test.cc"],
  deps = [
    ":transpose",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

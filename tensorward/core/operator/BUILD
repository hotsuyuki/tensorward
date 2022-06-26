load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "add",
  hdrs = ["add.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "div",
  hdrs = ["div.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "mul",
  hdrs = ["mul.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "neg",
  hdrs = ["neg.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "sub",
  hdrs = ["sub.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "add_test",
  srcs = ["test/add_test.cc"],
  deps = [
    ":add",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "div_test",
  srcs = ["test/div_test.cc"],
  deps = [
    ":div",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "mul_test",
  srcs = ["test/mul_test.cc"],
  deps = [
    ":mul",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "neg_test",
  srcs = ["test/neg_test.cc"],
  deps = [
    ":neg",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

cc_test(
  name = "sub_test",
  srcs = ["test/sub_test.cc"],
  deps = [
    ":sub",
    "//tensorward/util:numerical_gradient",
    "@com_google_googletest//:gtest_main",
    "@xtensor//:xtensor",
  ],
)

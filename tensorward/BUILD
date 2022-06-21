load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "core",
  hdrs = ["core.h"],
  deps = [
    "//tensorward/core:function",
    "//tensorward/core:tensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "function",
  hdrs = ["function.h"],
  deps = [
    "//tensorward/function:exp",
    "//tensorward/function:square",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "util",
  hdrs = ["util.h"],
  deps = [
    "//tensorward/util:numerical_gradient",
  ],
  visibility = ["//visibility:public"],
)

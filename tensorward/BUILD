load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "core",
  hdrs = ["core.h"],
  deps = [
    "//tensorward/core:config",
    "//tensorward/core:function",
    "//tensorward/core:tensor",
    "//tensorward/core/operator:add",
    "//tensorward/core/operator:div",
    "//tensorward/core/operator:mul",
    "//tensorward/core/operator:neg",
    "//tensorward/core/operator:sub",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "function",
  hdrs = ["function.h"],
  deps = [
    "//tensorward/function:broadcast_to",
    "//tensorward/function:exp",
    "//tensorward/function:linear",
    "//tensorward/function:matmul",
    "//tensorward/function:mean_squared_error",
    "//tensorward/function:pow",
    "//tensorward/function:reshape",
    "//tensorward/function:square",
    "//tensorward/function:sum",
    "//tensorward/function:sum_to",
    "//tensorward/function:transpose",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "util",
  hdrs = ["util.h"],
  deps = [
    "//tensorward/util:numerical_gradient",
    "//tensorward/util:xtensor_sum_to",
  ],
  visibility = ["//visibility:public"],
)

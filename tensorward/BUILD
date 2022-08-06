load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "core",
  hdrs = ["core.h"],
  deps = [
    "//tensorward/core:config",
    "//tensorward/core:function",
    "//tensorward/core:layer",
    "//tensorward/core:model",
    "//tensorward/core:parameter",
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
    "//tensorward/function:get_item",
    "//tensorward/function:linear",
    "//tensorward/function:matmul",
    "//tensorward/function:mean_squared_error",
    "//tensorward/function:pow",
    "//tensorward/function:reshape",
    "//tensorward/function:sigmoid",
    "//tensorward/function:softmax_cross_entropy_error",
    "//tensorward/function:square",
    "//tensorward/function:sum",
    "//tensorward/function:sum_to",
    "//tensorward/function:transpose",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "layer",
  hdrs = ["layer.h"],
  deps = [
    "//tensorward/layer:linear",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "model",
  hdrs = ["model.h"],
  deps = [
    "//tensorward/model:multi_layer_perceptron",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "optimizer",
  hdrs = ["optimizer.h"],
  deps = [
    "//tensorward/optimizer:momentum_stochastic_gradient_descent",
    "//tensorward/optimizer:stochastic_gradient_descent",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "util",
  hdrs = ["util.h"],
  deps = [
    "//tensorward/util:numerical_gradient",
    "//tensorward/util:xtensor_cross_entropy_error",
    "//tensorward/util:xtensor_softmax",
    "//tensorward/util:xtensor_sum_to",
  ],
  visibility = ["//visibility:public"],
)

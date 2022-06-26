load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    "//tensorward:core",
    "//tensorward:function",
    "@xtensor//:xtensor",
  ],
)

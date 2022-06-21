load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "xtensor",
  hdrs = glob(["include/xtensor/*.hpp"]),
  strip_include_prefix = "include/",
  deps = [
    "@xtl//:xtl",
  ],
  visibility = ["//visibility:public"],
)

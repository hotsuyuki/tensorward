load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
  name = "mnist_one_tenth",
  hdrs = [
    "mnist_one_tenth.h",
  ],
  deps = [
    ":mnist",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "mnist",
  hdrs = [
    "mnist.h",
  ],
  deps = [
    "//tensorward/core:dataset",
    "@curl//:curl",
    "@gzip-hpp//:gzip",
    "@xtensor//:xtensor",
  ],
  linkopts = [
    "-l curl",  # libcurl (URL data transfer library)
    "-l z",     # zlib (compression library)
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "spiral",
  hdrs = [
    "spiral.h",
  ],
  deps = [
    "//tensorward/core:dataset",
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

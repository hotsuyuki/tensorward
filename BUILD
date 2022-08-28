load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "curl",
  hdrs = glob(["include/curl/*.h"]),
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "gzip",
  hdrs = glob(["include/gzip/*.hpp"]),
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "xtensor-blas",
  hdrs = glob([
    "include/xflens/cxxblas/**/*.h",
    "include/xflens/cxxblas/**/*.tcc",
    "include/xflens/cxxblas/**/*.cxx",
    "include/xflens/cxxlapack/**/*.h",
    "include/xflens/cxxlapack/**/*.tcc",
    "include/xflens/cxxlapack/**/*.cxx",
    "include/xtensor-blas/*.hpp",
  ],
  exclude = [
    "include/xflens/cxxblas/netlib/",
    "include/xflens/cxxlapack/netlib/",
  ]),
  strip_include_prefix = "include/",
  deps = [
    "@xtensor//:xtensor",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "xtensor",
  hdrs = glob(["include/xtensor/*.hpp"]),
  strip_include_prefix = "include/",
  deps = [
    "@xtl//:xtl",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "xtl",
  hdrs = glob(["include/xtl/*.hpp"]),
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)

load("@rules_cc//cc:defs.bzl", "cc_library")

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

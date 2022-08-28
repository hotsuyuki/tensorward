load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
  name = "curl",
  hdrs = glob(["include/curl/*.h"]),
  strip_include_prefix = "include/",
  visibility = ["//visibility:public"],
)

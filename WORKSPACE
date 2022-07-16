load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "com_google_googletest",
  url = "https://github.com/google/googletest/archive/release-1.11.0.zip",
  sha256 = "353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a",
  strip_prefix = "googletest-release-1.11.0",
)

http_archive(
  name = "xtl",
  url = "https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.0.zip",
  sha256 = "575794966e4755ded190f22ce9dea7a3c2aee49cc5ce5e14fe1e54cd0fe60184",
  build_file = "@//:xtl.BUILD",
  strip_prefix = "xtl-0.7.0",
)

http_archive(
  name = "xtensor",
  url = "https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.0.zip",
  sha256 = "0823e26127fa387efa45b77c2d151ad38e0f9a490850729821f9a8ae399d0069",
  build_file = "@//:xtensor.BUILD",
  strip_prefix = "xtensor-0.24.0",
)

http_archive(
  name = "xtensor-blas",
  url = "https://github.com/xtensor-stack/xtensor-blas/archive/refs/tags/0.20.0.zip",
  sha256 = "e8a62c11c0fb912c3028d79879154296ab31c36b61eb3fac42d65aa859fd1f8a",
  build_file = "@//:xtensor-blas.BUILD",
  strip_prefix = "xtensor-blas-0.20.0",
)

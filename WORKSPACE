load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "com_google_googletest",
  url = "https://github.com/google/googletest/archive/release-1.11.0.zip",
  sha256 = "353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a",
  strip_prefix = "googletest-release-1.11.0",
)

# TODO: Move "curl.BUILD, gzip-hpp.BUILD, xtensor-blas.BUILD, xtensor.BUILD, xtl.BUILD" into one place "BUILD"

http_archive(
  name = "curl",
  url = "https://github.com/curl/curl/archive/refs/tags/curl-7_84_0.zip",
  sha256 = "e9d74b8586e0d2e6b45dc948bbe77525a1fa7f7c004ad5192f12e72c365e376e",
  build_file = "@//:curl.BUILD",
  strip_prefix = "curl-curl-7_84_0",
)

http_archive(
  name = "gzip-hpp",
  url = "https://github.com/mapbox/gzip-hpp/archive/refs/tags/v0.1.0.zip",
  sha256 = "e44c89ff6fa5ccb99411dad4a6c47fc84efab5bf13970032af5e67de7f6f09dc",
  build_file = "@//:gzip-hpp.BUILD",
  strip_prefix = "gzip-hpp-0.1.0",
)

# TODO: Change the order from "xtl, xtensor, xtensor-blas" to "xtensor-blas, xtensor, xtl" (alphabetically)

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

#pragma once

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <gzip/decompress.hpp>
#include <gzip/utils.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "tensorward/core/dataset.h"

namespace tensorward::dataset {

namespace {

// Reference: https://curl.se/libcurl/c/url2file.html
std::size_t WriteDataCallback(const void* ptr, const std::size_t size, const std::size_t nmemb, void* stream) {
  const std::size_t written = fwrite(ptr, size, nmemb, static_cast<FILE*>(stream));
  return written;
}

// Reference: https://curl.se/libcurl/c/url2file.html
void DownloadFileFromURLToPath(const char* input_url, const char* output_file_path) {
  std::cout << "Downloading a file from '" << input_url << "' to '" << output_file_path << "' ..." << std::endl;

  CURL* curl_handle;
  FILE* output_file;

  curl_global_init(CURL_GLOBAL_ALL);
  curl_handle = curl_easy_init();

  curl_easy_setopt(curl_handle, CURLOPT_URL, input_url);
  curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 1);
  curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 0);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteDataCallback);

  output_file = fopen(output_file_path, "wb");
  if (output_file) {
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, output_file);
    curl_easy_perform(curl_handle);
    fclose(output_file);
  }

  curl_easy_cleanup(curl_handle);
  curl_global_cleanup();

  std::cout << std::endl << std::endl;
}

std::stringstream DecompressFile(const std::filesystem::path& file_path) {
  const std::size_t file_size = std::filesystem::file_size(file_path);
  char* file_pointer = new char[file_size];

  std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
  ifs.read(file_pointer, file_size);
  ifs.close();

  assert(gzip::is_compressed(file_pointer, file_size));
  const std::string decompressed_data = gzip::decompress(file_pointer, file_size);
  delete[] file_pointer;

  return std::stringstream(decompressed_data);
}

int ReverseEndian(const int integer) {
  const std::byte byte0 = static_cast<std::byte>((integer >> 0) & 255);
  const std::byte byte1 = static_cast<std::byte>((integer >> 8) & 255);
  const std::byte byte2 = static_cast<std::byte>((integer >> 16) & 255);
  const std::byte byte3 = static_cast<std::byte>((integer >> 24) & 255);

  const int reversed_integer = (static_cast<int>(byte0) << 24) + (static_cast<int>(byte1) << 16) +
                               (static_cast<int>(byte2) << 8) + (static_cast<int>(byte3) << 0);

  return reversed_integer;
}

xt::xarray<float> ReadDataFile(const std::filesystem::path& data_file_path, const bool is_verbose = false) {
  std::stringstream ss = DecompressFile(data_file_path);

  int magic_number;
  ss.read((char*)&magic_number, sizeof(magic_number));
  magic_number = ReverseEndian(magic_number);

  int data_size;
  ss.read((char*)&data_size, sizeof(data_size));
  data_size = ReverseEndian(data_size);

  const int channel = 1;  // Because each image is gray-scale.

  int height;
  ss.read((char*)&height, sizeof(height));
  height = ReverseEndian(height);

  int width;
  ss.read((char*)&width, sizeof(width));
  width = ReverseEndian(width);

  if (is_verbose) {
    std::cout << "[ReadDataFile()] magic_number = " << magic_number << ", data_size = " << data_size
              << ", height = " << height << ", width = " << width << std::endl;
  }

  xt::xarray<float> data = xt::zeros<float>({data_size, channel, height, width});
  for (std::size_t i = 0; i < data_size; ++i) {
    xt::xarray<float> ith_data = xt::zeros<float>({channel, height, width});
    for (std::size_t h = 0; h < height; ++h) {
      for (std::size_t w = 0; w < width; ++w) {
        std::byte one_byte;
        ss.read((char*)&one_byte, sizeof(one_byte));
        ith_data(h, w) = static_cast<float>(one_byte);
      }
    }
    xt::view(data, i) = ith_data;
  }

  assert(data.dimension() == 4);  // {N, C, H, W}
  assert(data.shape(0) == data_size);
  assert(data.shape(1) == channel);
  assert(data.shape(2) == height);
  assert(data.shape(3) == width);

  return data;
}

xt::xarray<float> ReadLabelFile(const std::filesystem::path& label_file_path, const bool is_verbose = false) {
  std::stringstream ss = DecompressFile(label_file_path);

  int magic_number;
  ss.read((char*)&magic_number, sizeof(magic_number));
  magic_number = ReverseEndian(magic_number);

  int label_size;
  ss.read((char*)&label_size, sizeof(label_size));
  label_size = ReverseEndian(label_size);

  if (is_verbose) {
    std::cout << "[ReadLabelFile()] magic_number = " << magic_number << ", label_size = " << label_size << std::endl;
  }

  xt::xarray<float> label = xt::zeros<float>({label_size});
  for (std::size_t i = 0; i < label_size; ++i) {
    std::byte one_byte;
    ss.read((char*)&one_byte, sizeof(one_byte));
    label(i) = static_cast<float>(one_byte);
  }

  assert(label.dimension() == 1);  // {N}
  assert(label.shape(0) == label_size);

  return label;
}

}  // namespace

class Mnist : public core::Dataset {
 public:
  Mnist(const bool is_training_mode, const std::vector<core::TransformLambda>& data_transform_lambdas,
        const std::vector<core::TransformLambda>& label_transform_lambdas)
      : core::Dataset(is_training_mode, data_transform_lambdas, label_transform_lambdas, "Mnist") {
    Init();
  }

  ~Mnist() {}

  void Init() override {
    const std::filesystem::path data_file_name =
        is_training_mode_ ? "train-images-idx3-ubyte.gz" : "t10k-images-idx3-ubyte.gz";
    const std::filesystem::path label_file_name =
        is_training_mode_ ? "train-labels-idx1-ubyte.gz" : "t10k-labels-idx1-ubyte.gz";

    const std::filesystem::path base_url = "http://yann.lecun.com/exdb/mnist";
    const std::filesystem::path data_file_url = base_url / data_file_name;
    const std::filesystem::path label_file_url = base_url / label_file_name;

    const std::filesystem::path data_file_path = dataset_directory_path_ / data_file_name;
    const std::filesystem::path label_file_path = dataset_directory_path_ / label_file_name;

    if (!std::filesystem::exists(data_file_path)) {
      DownloadFileFromURLToPath(data_file_url.c_str(), data_file_path.c_str());
    }
    if (!std::filesystem::exists(label_file_path)) {
      DownloadFileFromURLToPath(label_file_url.c_str(), label_file_path.c_str());
    }

    data_ = ReadDataFile(data_file_path);
    label_ = ReadLabelFile(label_file_path);

    std::cout << "[Mnist::Init()] xt::view(data_, 0) =" << std::endl
              << xt::print_options::line_width(1000) << xt::view(data_, 0) << std::endl << std::endl;
    std::cout << "[Mnist::Init()] xt::view(label_, 0) =" << std::endl
              << xt::print_options::line_width(1000) << xt::view(label_, 0) << std::endl << std::endl;
  }
};

}  // namespace tensorward::dataset

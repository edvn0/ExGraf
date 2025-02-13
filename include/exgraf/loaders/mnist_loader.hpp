#pragma once

#include <armadillo>
#include <stdexcept>
#include <vector>
#include <zlib.h>

#include <array>
#include <future>

#include "exgraf/http/client.hpp"
#include "exgraf/model.hpp"
#include "exgraf/tensor.hpp"

#ifdef HAS_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

namespace ExGraf::MNIST {

inline auto decompress_gzip(const std::vector<unsigned char> &input)
    -> std::vector<unsigned char> {
  // Minimal in-memory zlib inflate for GZip data
  // If your data is not GZip, skip this step
  z_stream strm{};
  strm.avail_in = static_cast<uInt>(input.size());
  strm.next_in =
      reinterpret_cast<Bytef *>(const_cast<unsigned char *>(input.data()));
  if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
    throw std::runtime_error("Failed to init zlib for GZip.");
  }

  std::vector<unsigned char> output;
  output.resize(10 * 1024 * 1024); // 10MB temp buffer; adjust as needed

  int ret;
  do {
    strm.avail_out = static_cast<uInt>(output.size() - strm.total_out);
    strm.next_out = reinterpret_cast<Bytef *>(output.data() + strm.total_out);
    ret = inflate(&strm, Z_NO_FLUSH);
    if (ret == Z_STREAM_END)
      break;
    if (ret == Z_OK && strm.avail_out == 0) {
      // Increase buffer and continue
      output.resize(output.size() * 2);
    } else if (ret != Z_OK) {
      inflateEnd(&strm);
      throw std::runtime_error("zlib error during inflate.");
    }
  } while (true);

  inflateEnd(&strm);
  output.resize(strm.total_out);
  return output;
}

inline auto parse_idx_images(const std::vector<unsigned char> &buffer)
    -> arma::Mat<double> {
  if (buffer.size() < 16)
    throw std::runtime_error("Invalid IDX image file.");
  int magic =
      (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
  int num_images =
      (buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | buffer[7];
  int num_rows =
      (buffer[8] << 24) | (buffer[9] << 16) | (buffer[10] << 8) | buffer[11];
  int num_cols =
      (buffer[12] << 24) | (buffer[13] << 16) | (buffer[14] << 8) | buffer[15];

  if (magic != 2051)
    throw std::runtime_error("Not an IDX image file.");

  arma::Mat<double> data(num_images, num_rows * num_cols, arma::fill::zeros);
  std::size_t offset = 16;
  for (int i = 0; i < num_images; i++) {
    for (int p = 0; p < num_rows * num_cols; p++) {
      data(i, p) = buffer[offset++] / 255.0; // Scale [0..1]
    }
  }
  return data;
}

inline auto parse_idx_labels(const std::vector<unsigned char> &buffer)
    -> arma::Col<std::size_t> {
  if (buffer.size() < 8)
    throw std::runtime_error("Invalid IDX label file.");
  int magic =
      (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
  int num_items =
      (buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | buffer[7];

  if (magic != 2049)
    throw std::runtime_error("Not an IDX label file.");

  arma::Col<std::size_t> labels(num_items);
  std::size_t offset = 8;
  for (int i = 0; i < num_items; i++) {
    labels(i) = static_cast<std::size_t>(buffer[offset++]);
  }
  return labels;
}

inline auto load_mnist(const std::string &images_url,
                       const std::string &labels_url)
    -> std::pair<ExGraf::Tensor<double>, ExGraf::Tensor<double>> {
  ExGraf::Http::HttpClient client;

  std::array<std::future<ExGraf::Http::HttpResponse>, 2> futures{};
  futures[0] = std::async(std::launch::async, [&client, &images_url] {
    return client.get(images_url);
  });
  futures[1] = std::async(std::launch::async, [&client, &labels_url] {
    return client.get(labels_url);
  });

  for (auto &f : futures) {
    if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      throw std::runtime_error("HTTP request timed out.");
    }
  }

  auto img_res = futures[0].get();
  auto lbl_res = futures[1].get();

  std::vector<unsigned char> img_bytes(img_res.body.begin(),
                                       img_res.body.end());
  std::vector<unsigned char> lbl_bytes(lbl_res.body.begin(),
                                       lbl_res.body.end());

  std::array<std::future<std::vector<unsigned char>>, 2> decompress_futures{};
  decompress_futures[0] =
      std::async(std::launch::async, [bytes = std::move(img_bytes)] {
        return decompress_gzip(bytes);
      });
  decompress_futures[1] =
      std::async(std::launch::async, [bytes = std::move(lbl_bytes)] {
        return decompress_gzip(bytes);
      });

  for (auto &f : decompress_futures) {
    if (f.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      throw std::runtime_error("Decompression timed out.");
    }
  }

  std::tuple<std::future<ExGraf::Tensor<double>>,
             std::future<ExGraf::Tensor<double>>>
      parse_futures{};
  std::get<0>(parse_futures) = std::async(
      std::launch::async, [f = std::move(decompress_futures[0].get())] {
        auto parsed = parse_idx_images(f);
        return ExGraf::Tensor<double>(parsed);
      });
  std::get<1>(parse_futures) = std::async(
      std::launch::async, [f = std::move(decompress_futures[1].get())] {
        auto parsed = parse_idx_labels(f);
        return ExGraf::Model<double>::to_one_hot(parsed, 10);
      });

  return {std::get<0>(parse_futures).get(), std::get<1>(parse_futures).get()};
}

#ifdef HAS_TASKFLOW
inline auto load_mnist_taskflow(const std::string &images_url,
                                const std::string &labels_url)
    -> std::pair<ExGraf::Tensor<double>, ExGraf::Tensor<double>> {
  ExGraf::Http::HttpClient client;
  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<unsigned char> img_bytes, lbl_bytes;
  std::vector<unsigned char> img_decompressed, lbl_decompressed;
  ExGraf::Tensor<double> img_tensor, lbl_tensor;

  auto fetch_images = taskflow.emplace([&] {
    auto res = client.get(images_url);
    img_bytes.assign(res.body.begin(), res.body.end());
  });
  auto fetch_labels = taskflow.emplace([&] {
    auto res = client.get(labels_url);
    lbl_bytes.assign(res.body.begin(), res.body.end());
  });
  auto decompress_images =
      taskflow.emplace([&] { img_decompressed = decompress_gzip(img_bytes); });
  auto decompress_labels =
      taskflow.emplace([&] { lbl_decompressed = decompress_gzip(lbl_bytes); });
  auto parse_images = taskflow.emplace([&] {
    auto parsed = parse_idx_images(img_decompressed);
    img_tensor = ExGraf::Tensor<double>(parsed);
  });
  auto parse_labels = taskflow.emplace([&] {
    auto parsed = parse_idx_labels(lbl_decompressed);
    lbl_tensor = ExGraf::Model<double>::to_one_hot(parsed, 10);
  });

  fetch_images.precede(decompress_images);
  fetch_labels.precede(decompress_labels);
  decompress_images.precede(parse_images);
  decompress_labels.precede(parse_labels);

  static bool first = true;
  if (first) {
    taskflow.dump(std::cout);
    first = false;
  }

  executor.run(taskflow).wait();
  return {img_tensor, lbl_tensor};
}
#endif

} // namespace ExGraf::MNIST

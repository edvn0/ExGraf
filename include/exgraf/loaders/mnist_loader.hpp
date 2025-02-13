#pragma once

#include <armadillo>
#include <stdexcept>
#include <vector>
#include <zlib.h>

#include "exgraf/http/client.hpp"
#include "exgraf/model.hpp"
#include "exgraf/tensor.hpp"

#include <taskflow/taskflow.hpp>

namespace ExGraf::MNIST {

inline auto decompress_gzip(const std::vector<unsigned char> &input)
		-> std::vector<unsigned char> {
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
		img_tensor = Tensor<double>(parsed);
	});
	auto parse_labels = taskflow.emplace([&] {
		auto parsed = parse_idx_labels(lbl_decompressed);
		lbl_tensor = Model<double>::to_one_hot(parsed, 10);
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

} // namespace ExGraf::MNIST

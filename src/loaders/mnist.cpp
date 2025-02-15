#include "exgraf/loaders/mnist.hpp"

#include <armadillo>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <vector>
#include <zlib.h>

namespace ExGraf::MNIST {

namespace detail {

constexpr std::uint16_t IMAGE_MAGIC = 2051;
constexpr std::uint16_t LABEL_MAGIC = 2049;

// Function to parse big-endian integers from byte array
auto parse_int(const std::vector<std::uint8_t> &data, std::size_t offset)
		-> std::int32_t {
	return (static_cast<std::uint32_t>(data[offset]) << 24) |
				 (static_cast<std::uint32_t>(data[offset + 1]) << 16) |
				 (static_cast<std::uint32_t>(data[offset + 2]) << 8) |
				 static_cast<std::uint32_t>(data[offset + 3]);
}

// Function to decompress gzip data
auto decompress_gzip(const std::vector<std::uint8_t> &compressed_data)
		-> std::vector<std::uint8_t> {
	z_stream strm{};
	strm.next_in = const_cast<Bytef *>(compressed_data.data());
	strm.avail_in = static_cast<uInt>(compressed_data.size());

	if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
		throw std::runtime_error("Failed to initialize zlib for decompression");
	}

	std::vector<std::uint8_t> decompressed_data;
	std::array<std::uint8_t, 8192> buffer;

	int ret;
	do {
		strm.next_out = buffer.data();
		strm.avail_out = static_cast<uInt>(buffer.size());
		ret = inflate(&strm, Z_NO_FLUSH);

		if (decompressed_data.size() < strm.total_out) {
			decompressed_data.insert(decompressed_data.end(), buffer.data(),
															 buffer.data() +
																	 (strm.total_out - decompressed_data.size()));
		}
	} while (ret == Z_OK);

	inflateEnd(&strm);

	if (ret != Z_STREAM_END) {
		throw std::runtime_error("Failed to decompress gzip data");
	}

	return decompressed_data;
}

auto decode_images(const std::vector<std::uint8_t> &data) -> arma::Mat<double> {
	if (const auto parsed = parse_int(data, 0); parsed != IMAGE_MAGIC) {
		throw std::runtime_error("Invalid MNIST image file magic number");
	}
	std::uint32_t num_images = static_cast<std::uint32_t>(parse_int(data, 4));
	std::uint32_t rows = static_cast<std::uint32_t>(parse_int(data, 8));
	std::uint32_t cols = static_cast<std::uint32_t>(parse_int(data, 12));

	arma::Mat<double> images(rows * cols, num_images);
	for (std::uint32_t i = 0; i < num_images; ++i) {
		for (std::uint32_t j = 0; j < rows * cols; ++j) {
			images(j, i) =
					data[16 + i * rows * cols + j] / 255.0; // Normalize to [0, 1]
		}
	}
	return images;
}

auto decode_labels(const std::vector<std::uint8_t> &data) -> arma::Mat<double> {
	if (const auto parsed = parse_int(data, 0); parsed != LABEL_MAGIC) {
		throw std::runtime_error("Invalid MNIST label file magic number");
	}
	auto num_labels = static_cast<std::uint32_t>(parse_int(data, 4));

	arma::Mat<double> labels(10, num_labels, arma::fill::zeros);
	for (std::uint32_t i = 0; i < num_labels; ++i) {
		labels(static_cast<int>(data[8 + i]), i) = 1.0;
	}
	return labels;
}

} // namespace detail

} // namespace ExGraf::MNIST

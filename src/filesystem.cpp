#include "exgraf/loaders/mnist.hpp"

#include <armadillo>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <taskflow/taskflow.hpp>
#include <vector>
#include <zlib.h>

namespace ExGraf::FS {

constexpr std::uint16_t IMAGE_MAGIC = 2051;
constexpr std::uint16_t LABEL_MAGIC = 2049;

auto parse_int(const std::vector<std::uint8_t> &data, std::size_t offset)
		-> std::int32_t {

	static constexpr auto shift_and_cast = [](auto &C, auto off, auto shift) {
		return static_cast<std::int32_t>(C.at(off) << shift);
	};

	auto with_data = [&d = data](auto off, std::int32_t shift = (0)) {
		return shift_and_cast(d, off, shift);
	};

	return with_data(offset, 24) | with_data(offset + 1, 16) |
				 with_data(offset + 2, 8) | with_data(offset + 3);
}

auto decompress_gzip(const std::span<const std::uint8_t> compressed_data)
		-> std::vector<std::uint8_t> {
	z_stream strm{};
	strm.next_in = const_cast<std::uint8_t *>(compressed_data.data());
	strm.avail_in = static_cast<uInt>(compressed_data.size());

	if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
		throw std::runtime_error("Failed to initialize zlib for decompression");
	}

	std::vector<std::uint8_t> decompressed_data;
	std::array<std::uint8_t, 8192> buffer{};

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

auto decode_images(const std::vector<std::uint8_t> &data,
									 arma::Mat<double> &images) -> bool {
	if (const auto parsed = parse_int(data, 0); parsed != IMAGE_MAGIC) {
		return false;
	}
	std::uint32_t num_images = static_cast<std::uint32_t>(parse_int(data, 4));
	std::uint32_t rows = static_cast<std::uint32_t>(parse_int(data, 8));
	std::uint32_t cols = static_cast<std::uint32_t>(parse_int(data, 12));

	images.resize(arma::SizeMat{
			cols * rows,
			num_images,
	});
	for (std::uint32_t i = 0; i < num_images; ++i) {
		for (std::uint32_t j = 0; j < rows * cols; ++j) {
			images(j, i) =
					data[16 + i * rows * cols + j] / 255.0; // Normalize to [0, 1]
		}
	}

	return true;
}

auto decode_labels(const std::vector<std::uint8_t> &data,
									 arma::Mat<double> &labels) -> bool {
	if (const auto parsed = parse_int(data, 0); parsed != LABEL_MAGIC) {
		return false;
	}
	auto num_labels = static_cast<std::uint32_t>(parse_int(data, 4));

	labels.resize(arma::SizeMat{10, num_labels});
	for (std::uint32_t i = 0; i < num_labels; ++i) {
		labels(static_cast<unsigned long long>(data[8 + i]), i) = 1.0;
	}

	return true;
}

auto read_file(const std::string &path) -> std::vector<std::uint8_t> {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("Failed to open file: " + path);
	}

	std::vector<std::uint8_t> buffer;
	buffer.assign(std::istreambuf_iterator<char>(file),
								std::istreambuf_iterator<char>());

	return buffer;
}

} // namespace ExGraf::FS

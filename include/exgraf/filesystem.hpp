#pragma once

#include <armadillo>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ExGraf::FS {

using ByteBuffer = std::vector<std::uint8_t>;

auto parse_int(const ByteBuffer &, std::size_t) -> std::int32_t;
auto decode_images(const ByteBuffer &, arma::Mat<double> &) -> bool;
auto decode_labels(const ByteBuffer &, arma::Mat<double> &) -> bool;
auto read_file(const std::string &) -> ByteBuffer;
auto decompress_gzip(std::span<const std::uint8_t>) -> ByteBuffer;
inline auto decompress_gzip(const std::vector<std::uint8_t> &vec)
		-> ByteBuffer {
	return decompress_gzip(std::span(vec));
}

} // namespace ExGraf::FS

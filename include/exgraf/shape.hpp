#pragma once

#include <armadillo>
#include <numeric>

namespace ExGraf {

class Shape {
public:
	Shape() = default;
	template <std::integral T>
	explicit Shape(std::initializer_list<T> ds) : dimensions(ds) {}
	explicit Shape(const arma::SizeMat &size)
			: dimensions{
						static_cast<unsigned char>(size.n_rows),
						static_cast<unsigned char>(size.n_cols),
				} {}
	auto total_elements() -> std::size_t {
		return std::accumulate(dimensions.begin(), dimensions.end(), std::size_t{1},
													 std::multiplies<std::size_t>());
	}
	auto operator==(const Shape &other) const -> bool = default;
	auto dims() const -> const auto & { return dimensions; }

	auto rows() const { return dimensions[0]; }
	auto cols() const { return dimensions[1]; }

	auto operator[](std::size_t i) const { return dimensions[i]; }

private:
	std::vector<std::uint8_t> dimensions{};
};

} // namespace ExGraf

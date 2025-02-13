#pragma once

#include <armadillo>
#include <numeric>

namespace ExGraf {

class Shape {
public:
	Shape() = default;
	Shape(std::initializer_list<std::size_t> ds) : dimensions(ds) {}
	Shape(const arma::SizeMat &size) : dimensions{size.n_rows, size.n_cols} {}
	auto total_elements() -> std::size_t {
		return std::accumulate(dimensions.begin(), dimensions.end(), std::size_t{1},
													 std::multiplies<std::size_t>());
	}
	auto operator==(const Shape &other) const -> bool {
		return dimensions == other.dimensions;
	}
	auto dims() const -> const auto & { return dimensions; }

private:
	std::vector<std::size_t> dimensions{};
};

} // namespace ExGraf

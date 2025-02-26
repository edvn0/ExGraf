#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"

#include <cstdint>

namespace ExGraf {

namespace detail {

static std::mutex allocation_mutex;
static std::unordered_map<void *, bool> allocations;

inline auto report_leaks() -> void {
	std::lock_guard lock(allocation_mutex);
	for (auto &&[ptr, could] : allocations) {
		if (!could) {
			info("Memory leak at {}\n", static_cast<const void *>(ptr));
		}
	}
}

template <typename T> struct TrackingAllocator {
	using value_type = T;
	TrackingAllocator() = default;
	template <typename U>
	explicit TrackingAllocator(const TrackingAllocator<U> &) {}

	auto allocate(std::size_t n) -> T * {
		auto ptr = static_cast<T *>(::operator new(n * sizeof(T)));
		{
			std::lock_guard lock(allocation_mutex);
			allocations[ptr] = false;
		}
		return ptr;
	}

	auto deallocate(T *ptr, std::size_t) -> void {
		{
			std::lock_guard lock(allocation_mutex);
			auto it = allocations.find(ptr);
			if (it != allocations.end()) {
				if (it->second) {
					info("Double free at {}\n", static_cast<const void *>(ptr));
				} else {
					it->second = true;
				}
			}
		}
		::operator delete(ptr);
	}
};

template <typename T, typename U>
inline bool operator==(const TrackingAllocator<T> &,
											 const TrackingAllocator<U> &) {
	return true;
}

template <typename T, typename U>
inline bool operator!=(const TrackingAllocator<T> &,
											 const TrackingAllocator<U> &) {
	return false;
}
} // namespace detail

struct InvalidInputShapeError : std::logic_error {
	using std::logic_error::logic_error;
};

struct UnsupportedOperationError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

enum class ActivationFunction : std::uint8_t {
	ReLU,
	Tanh,
};
enum class LossFunction : std::uint8_t {
	MeanSquaredError,
	CrossEntropy,
};
enum class OutputActivationFunction : std::uint8_t {
	Softmax,
	ReLU,
};
enum class OptimizerType : std::uint8_t {
	SGD,

	ADAM,
};

template <AllowedTypes T>
static constexpr auto randn_matrix(std::uint32_t rows, std::uint32_t cols) {
	arma::Mat<T> matrix(cols, rows, arma::fill::randn);
	return matrix;
}

template <AllowedTypes T>
arma::Mat<T> initialize_weights(std::uint32_t input_size,
																std::uint32_t output_size,
																ActivationFunction activation) {
	if (activation == ActivationFunction::ReLU) {
		return randn_matrix<T>(input_size, output_size) *
					 std::sqrt(2.0 / input_size);
	} else {
		return randn_matrix<T>(input_size, output_size) *
					 std::sqrt(2.0 / (input_size + output_size));
	}
}
} // namespace ExGraf

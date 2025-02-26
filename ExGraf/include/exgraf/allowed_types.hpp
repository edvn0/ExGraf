#pragma once

#include <cstdint>
#include <type_traits>

namespace ExGraf {

#define EXGRAF_ALLOWED_TYPES                                                   \
	X(float)                                                                     \
	X(double)

template <typename T>
concept AllowedTypes = []() {
	bool result = false;
#define X(type) result |= std::is_same_v<T, type>;
	EXGRAF_ALLOWED_TYPES
#undef X
	return result;
}();

} // namespace ExGraf

#pragma once

#include <concepts>
#include <type_traits>

namespace ExGraf {

template <typename T>
concept AllowedTypes = std::is_same_v<T, float> || std::is_same_v<T, double>;

#define EXGRAF_ALLOWED_TYPES                                                   \
  X(float)                                                                     \
  X(double)

} // namespace ExGraf
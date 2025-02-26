#include "exgraf/exgraf_pch.hpp"

#include <cstdlib>

#include "exgraf/logger.hpp"

namespace ExGraf {

auto get_from_environment(std::string_view key) -> std::string {
	if (const auto value = std::getenv(key.data()); value != nullptr) {
		return value;
	}
	return {};
}

} // namespace ExGraf

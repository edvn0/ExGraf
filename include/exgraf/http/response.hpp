#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ExGraf::Http {

enum class HttpMethod : std::uint8_t { GET, POST, PUT, DELETE };

enum class ErrorCode : std::uint8_t {
	OK,
	UNKNOWN_ERROR,
};

struct HttpError {
	ErrorCode code;
	std::string message;
};

struct HttpResponse {
	HttpMethod method;
	int status_code;
	std::string body;
	bool success;
	std::optional<HttpError> error;
};

} // namespace ExGraf::Http

#pragma once

#include <string>

namespace ExGraf::Http {

struct HttpResponse {
	int status_code;
	std::string body;
	bool success;
	std::string error_message;
};

} // namespace ExGraf::Http

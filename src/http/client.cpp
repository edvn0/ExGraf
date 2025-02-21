#include "exgraf/http/client.hpp"
#include "exgraf/http/response.hpp"

#include <cpr/cpr.h>

#include <taskflow/core/executor.hpp>
#include <taskflow/taskflow.hpp>
#include <utility>

namespace ExGraf::Http {

static constexpr auto to_int(long int value) -> int {
	return static_cast<int>(value);
}

static constexpr auto to_error_code(cpr::ErrorCode value) -> ErrorCode {
	switch (value) {
	case cpr::ErrorCode::OK:
		return ErrorCode::OK;
	default:
		return ErrorCode::UNKNOWN_ERROR;
	}
}

static constexpr auto create_error_maybe(cpr::Error error)
		-> std::optional<HttpError> {
	if (error.code == cpr::ErrorCode::OK)
		return std::nullopt;
	return HttpError{
			.code = to_error_code(error.code),
			.message = error.message,
	};
}

auto HttpClient::get(const std::string &endpoint) const -> HttpResponse {
	auto r = cpr::Get(cpr::Url{
			base_url + endpoint,
	});
	return HttpResponse{
			.method = HttpMethod::GET,
			.status_code = to_int(r.status_code),
			.body = r.text,
			.success = r.status_code == 200,
			.error = create_error_maybe(r.error),
	};
}

auto HttpClient::get(std::string_view endpoint) const -> HttpResponse {
	auto r = cpr::Get(cpr::Url{
			base_url + std::string(endpoint),
	});
	return HttpResponse{
			.method = HttpMethod::GET,
			.status_code = to_int(r.status_code),
			.body = r.text,
			.success = r.status_code == 200,
			.error = create_error_maybe(r.error),
	};
}

auto HttpClient::post(const std::string &endpoint,
											const std::string &payload) const -> HttpResponse {
	auto r = cpr::Post(
			cpr::Url{
					base_url + endpoint,
			},
			cpr::Body{
					payload,
			});
	return HttpResponse{
			.method = HttpMethod::POST,
			.status_code = to_int(r.status_code),
			.body = r.text,
			.success = r.status_code == 200,
			.error = create_error_maybe(r.error),
	};
}

auto HttpClient::put(const std::string &endpoint,
										 const std::string &payload) const -> HttpResponse {
	auto r = cpr::Put(
			cpr::Url{
					base_url + endpoint,
			},
			cpr::Body{
					payload,
			});
	return HttpResponse{
			.method = HttpMethod::PUT,
			.status_code = to_int(r.status_code),
			.body = r.text,
			.success = r.status_code == 200,
			.error = create_error_maybe(r.error),
	};
}

auto HttpClient::del(const std::string &endpoint) const -> HttpResponse {
	auto r = cpr::Delete(cpr::Url{
			base_url + endpoint,
	});
	return HttpResponse{
			.method = HttpMethod::DELETE,
			.status_code = to_int(r.status_code),
			.body = r.text,
			.success = r.status_code == 200,
			.error = create_error_maybe(r.error),
	};
}

auto MultithreadedDownloadClient::download_span(
		std::span<const std::string_view> urls) const -> std::vector<HttpResponse> {
	static tf::Executor executor;
	tf::Taskflow taskflow;
	std::vector<HttpResponse> responses(urls.size());
	for (std::size_t url_index = 0; url_index < urls.size(); ++url_index) {
		taskflow.emplace([&responses, url_index, this, url = urls[url_index]] {
			responses.at(url_index) = client.get(url);
		});
	}
	executor.run(taskflow).get();
	return responses;
}

} // namespace ExGraf::Http

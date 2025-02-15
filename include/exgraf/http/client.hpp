#pragma once

#include "exgraf/http/response.hpp"

#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace ExGraf::Http {

class HttpClient {
public:
	explicit HttpClient(const std::string &b = "") : base_url(b) {}

	auto get(const std::string &endpoint) const -> HttpResponse;
	auto get(std::string_view endpoint) const -> HttpResponse;
	auto post(const std::string &endpoint, const std::string &payload) const
			-> HttpResponse;
	auto put(const std::string &endpoint, const std::string &payload) const
			-> HttpResponse;
	auto del(const std::string &endpoint) const -> HttpResponse;

private:
	std::string base_url;
};

class MultithreadedDownloadClient {
public:
	explicit MultithreadedDownloadClient(const HttpClient &c) : client(c) {}

	template <std::ranges::contiguous_range R>
	auto download(R &&urls) const -> std::vector<HttpResponse> {
		return download_span(std::span(std::forward<R>(urls)));
	}

private:
	HttpClient client;

	auto download_span(const std::span<const std::string_view> urls) const
			-> std::vector<HttpResponse>;
};

} // namespace ExGraf::Http

#pragma once

#include "exgraf/http/response.hpp"

#include <string>

namespace ExGraf::Http {

class HttpClient {
public:
  explicit HttpClient(const std::string &base_url = "") : base_url(base_url) {}

  auto get(const std::string &endpoint) -> HttpResponse;
  auto post(const std::string &endpoint, const std::string &payload)
      -> HttpResponse;
  auto put(const std::string &endpoint, const std::string &payload)
      -> HttpResponse;
  auto del(const std::string &endpoint) -> HttpResponse;

private:
  std::string base_url;
};

} // namespace ExGraf::Http
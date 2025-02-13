#include "exgraf/http/client.hpp"

#include <cpr/cpr.h>

namespace ExGraf::Http {

static constexpr auto to_int(long int value) -> int {
  return static_cast<int>(value);
}

auto HttpClient::get(const std::string &endpoint) -> HttpResponse {
  auto r = cpr::Get(cpr::Url{base_url + endpoint});
  return HttpResponse{
      to_int(r.status_code),
      r.text,
      r.status_code == 200,
      r.error.message,
  };
}

auto HttpClient::post(const std::string &endpoint, const std::string &payload)
    -> HttpResponse {
  auto r = cpr::Post(cpr::Url{base_url + endpoint}, cpr::Body{payload});
  return HttpResponse{
      to_int(r.status_code),
      r.text,
      r.status_code == 200,
      r.error.message,
  };
}

auto HttpClient::put(const std::string &endpoint, const std::string &payload)
    -> HttpResponse {
  auto r = cpr::Put(cpr::Url{base_url + endpoint}, cpr::Body{payload});
  return HttpResponse{
      to_int(r.status_code),
      r.text,
      r.status_code == 200,
      r.error.message,
  };
}

auto HttpClient::del(const std::string &endpoint) -> HttpResponse {
  auto r = cpr::Delete(cpr::Url{base_url + endpoint});
  return HttpResponse{
      to_int(r.status_code),
      r.text,
      r.status_code == 200,
      r.error.message,
  };
}

} // namespace ExGraf::Http
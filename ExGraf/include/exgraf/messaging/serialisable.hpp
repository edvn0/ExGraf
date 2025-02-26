#pragma once

#include <boost/json/serialize.hpp>

namespace ExGraf::Messaging {

template <typename T> struct Serializer {
	static auto to_json(const T &obj) -> std::string {
		return boost::json::serialize(obj);
	}
};

} // namespace ExGraf::Messaging

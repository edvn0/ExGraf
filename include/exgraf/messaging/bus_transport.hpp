#pragma once

#include <string_view>

namespace ExGraf::Messaging {

template <typename Derived> class BusTransport {
public:
	void send(const std::string_view message) {
		static_cast<Derived *>(this)->send_impl(message);
	}
};

} // namespace ExGraf::Messaging

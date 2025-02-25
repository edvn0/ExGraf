#pragma once

#include "exgraf/messaging/metrics_logger_base.hpp"

#include <string_view>

namespace ExGraf::Messaging {

template <typename Derived> class BusTransport {
public:
	void send(const Messaging::MessageTo &message) {
		static_cast<Derived *>(this)->send_impl(message);
	}
};

} // namespace ExGraf::Messaging

#pragma once

#include "exgraf/visualisation/metrics_logger_base.hpp"

#include <string_view>

namespace ExGraf::Messaging {

template <typename Derived> class BusTransport {
public:
	void send(const UI::MessageTo &message) {
		static_cast<Derived *>(this)->send_impl(message);
	}
};

} // namespace ExGraf::Messaging

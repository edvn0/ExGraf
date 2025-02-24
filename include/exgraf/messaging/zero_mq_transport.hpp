#pragma once

#include "exgraf/messaging/bus_transport.hpp"

#include <memory>

namespace ExGraf::Messaging {

class ZeroMQTransport : public BusTransport<ZeroMQTransport> {
public:
	explicit ZeroMQTransport(const std::string &);
	~ZeroMQTransport();

	void send_impl(const UI::MessageTo &);

private:
	class ZeroMQTransportImpl;
	std::unique_ptr<ZeroMQTransportImpl> impl;
};

} // namespace ExGraf::Messaging

#pragma once

#include "exgraf/messaging/bus_transport.hpp"

#include <memory>

namespace ExGraf::Messaging {

class RabbitMQTransport : public BusTransport<RabbitMQTransport> {
public:
	explicit RabbitMQTransport(const std::string &);
	~RabbitMQTransport();

	void send_impl(const std::string_view);
	auto shutdown() -> void;

private:
	class RabbitMQTransportImpl;
	std::unique_ptr<RabbitMQTransportImpl> impl;
};

} // namespace ExGraf::Messaging

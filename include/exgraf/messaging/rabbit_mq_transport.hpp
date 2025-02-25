#pragma once

#include "exgraf/messaging/bus_transport.hpp"
#include "exgraf/messaging/metrics_logger_base.hpp"

#include <memory>

namespace ExGraf::Messaging {

class RabbitMQTransport : public BusTransport<RabbitMQTransport> {
public:
	explicit RabbitMQTransport(const std::string &);
	~RabbitMQTransport();

	void send_impl(const Messaging::MessageTo &);
	auto shutdown() -> void;
	auto wait_for_connection() const -> void;

private:
	class RabbitMQTransportImpl;
	std::unique_ptr<RabbitMQTransportImpl> impl;
};

} // namespace ExGraf::Messaging

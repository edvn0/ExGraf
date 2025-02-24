#include "exgraf/exgraf_pch.hpp"

#include "exgraf/messaging/zero_mq_transport.hpp"

#include <zmq.hpp>

namespace ExGraf::Messaging {

class ZeroMQTransport::ZeroMQTransportImpl {
public:
	explicit ZeroMQTransportImpl(const std::string &address)
			: context_(1), socket_(context_, zmq::socket_type::pub) {
		socket_.bind(address);
	}

	void send(const UI::MessageTo &message) {
		zmq::message_t zmq_message(message.message);
		socket_.send(zmq_message, zmq::send_flags::none);
	}

private:
	zmq::context_t context_;
	zmq::socket_t socket_;
};

ZeroMQTransport::ZeroMQTransport(const std::string &address)
		: impl(std::make_unique<ZeroMQTransportImpl>(address)) {}

ZeroMQTransport::~ZeroMQTransport() = default;

void ZeroMQTransport::send_impl(const UI::MessageTo &message) {
	impl->send(message);
}

} // namespace ExGraf::Messaging

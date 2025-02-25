#include "exgraf/exgraf_pch.hpp"

#include "exgraf/messaging/rabbit_mq_transport.hpp"

#include "exgraf/logger.hpp"

#include <amqpcpp.h>
#include <amqpcpp/address.h>
#include <amqpcpp/libboostasio.h>
#include <boost/asio.hpp>
#include <chrono>
#include <string_view>

namespace ExGraf::Messaging {

class RabbitMQTransport::RabbitMQTransportImpl {
public:
	explicit RabbitMQTransportImpl(std::string const &conn)
			: work_guard{boost::asio::make_work_guard(io_ctx_)}, handler{io_ctx_},
				connection{&handler, AMQP::Address(conn)}, channel{&connection} {
		channel.onReady([this, addr = AMQP::Address{conn}]() {
			channel.declareQueue("metrics", AMQP::durable);
			channel.declareExchange("metrics", AMQP::fanout, AMQP::durable);
			channel.bindQueue("metrics", "metrics", "metrics");
			is_connected = true;

			info("Connected to RabbitMQ at {}:{}", addr.hostname(), addr.port());
		});
		channel.onError([](char const *msg) { error("Channel error: {}", msg); });
		thread = std::jthread([this]() { io_ctx_.run(); });
	}

	~RabbitMQTransportImpl() {
		try {
			close_transport();
		} catch (std::exception const &e) {
			error("Exception in ~RabbitMQTransportImpl: {}", e.what());
		}
	}

	static constexpr auto to_rabbitmq_exchange =
			[](const Messaging::Outbox outbox) {
				switch (outbox) {
				case Messaging::Outbox::Metrics:
					return "metrics";
				case Messaging::Outbox::ModelConfiguration:
					return "model_configuration";
				}
				return "unknown";
			};

	auto send_impl(const Messaging::MessageTo &message) -> void {
		if (is_connected) {
			AMQP::Envelope envelope(message.message.data(), message.message.size());
			auto exchange = to_rabbitmq_exchange(message.outbox);
			auto routing_key = exchange;
			channel.publish(exchange, routing_key, envelope);
		}
	}

	auto close_transport() -> void {
		if (!closed) {
			if (is_connected) {
				channel.close();
				connection.close();
				boost::asio::steady_timer timer(io_ctx_);
				timer.expires_after(std::chrono::milliseconds(200));
				timer.async_wait([](auto) {});
				io_ctx_.run_for(std::chrono::milliseconds(250));
			}
			work_guard.reset();
			io_ctx_.stop();
			if (thread.joinable())
				thread.join();
			closed = true;
		}
	}

	auto wait_for_connection() const -> void {
		while (!is_connected) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

private:
	boost::asio::io_context io_ctx_;
	boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
			work_guard;
	AMQP::LibBoostAsioHandler handler;
	AMQP::TcpConnection connection;
	AMQP::TcpChannel channel;
	bool is_connected{false};
	bool closed{false};
	std::jthread thread;
};

RabbitMQTransport::RabbitMQTransport(std::string const &conn_string)
		: impl(std::make_unique<RabbitMQTransportImpl>(conn_string)) {}

RabbitMQTransport::~RabbitMQTransport() = default;

auto RabbitMQTransport::send_impl(const Messaging::MessageTo &message) -> void {
	impl->send_impl(message);
}

auto RabbitMQTransport::shutdown() -> void { impl->close_transport(); }

auto RabbitMQTransport::wait_for_connection() const -> void {
	info("Waiting for connection to RabbitMQ");
	impl->wait_for_connection();
	info("Connected to RabbitMQ");
}

} // namespace ExGraf::Messaging

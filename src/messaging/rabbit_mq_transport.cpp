#include "exgraf/exgraf_pch.hpp"

#include "exgraf/messaging/rabbit_mq_transport.hpp"

#include "exgraf/logger.hpp"

#include <amqpcpp.h>
#include <amqpcpp/address.h>
#include <amqpcpp/libboostasio.h>
#include <amqpcpp/linux_tcp/tcpchannel.h>
#include <boost/asio.hpp>
#include <chrono>
#include <string_view>

namespace ExGraf::Messaging {

class RabbitMQTransport::RabbitMQTransportImpl {
public:
	explicit RabbitMQTransportImpl(std::string const &conn)
			: work_guard{boost::asio::make_work_guard(io_ctx_)}, handler{io_ctx_},
				connection_address(AMQP::Address{conn}) {

		setup_channel_handlers();
		last_connect_attempt = std::chrono::steady_clock::now();
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
		if (!is_connected)
			return;

		AMQP::Envelope envelope(message.message.data(), message.message.size());
		auto exchange = to_rabbitmq_exchange(message.outbox);
		auto routing_key = exchange;
		channel->publish(exchange, routing_key, envelope);
	}

	auto close_transport() -> void {
		if (!closed) {
			if (is_connected) {
				channel->close();
				connection->close();
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

	auto wait_for_connection() -> void {
		unsigned int retry_count = 0;
		const unsigned int max_retries = 5;
		const auto retry_delay = std::chrono::seconds(10);

		while (!is_connected && retry_count < max_retries) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			if (!is_connected &&
					std::chrono::steady_clock::now() - last_connect_attempt >
							retry_delay) {

				info("Connection attempt {} failed, retrying...", retry_count + 1);

				if (channel_created) {
					channel->close();
					connection->close();
				}

				connection.reset(new AMQP::TcpConnection(&handler, connection_address));
				channel.reset(new AMQP::TcpChannel(connection.get()));

				setup_channel_handlers();

				last_connect_attempt = std::chrono::steady_clock::now();
				retry_count++;
			}
		}

		if (!is_connected) {
			error("Failed to connect to RabbitMQ after {} attempts", max_retries);
			throw std::runtime_error("Failed to connect to RabbitMQ");
		}
	}

private:
	boost::asio::io_context io_ctx_;
	boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
			work_guard;
	AMQP::LibBoostAsioHandler handler;
	std::unique_ptr<AMQP::TcpConnection> connection;
	std::unique_ptr<AMQP::TcpChannel> channel;
	bool is_connected{false};
	bool closed{false};
	std::jthread thread;
	std::chrono::steady_clock::time_point last_connect_attempt{
			std::chrono::steady_clock::now()};
	bool channel_created{true};
	AMQP::Address connection_address;

	void setup_channel_handlers() {
		connection =
				std::make_unique<AMQP::TcpConnection>(&handler, connection_address);
		channel = std::make_unique<AMQP::TcpChannel>(connection.get());

		channel->onReady([this]() {
			channel->declareQueue("metrics", AMQP::durable);
			channel->declareExchange("metrics", AMQP::fanout, AMQP::durable);
			channel->bindQueue("metrics", "metrics", "metrics");
			is_connected = true;
			info("Connected to RabbitMQ at {}:{}", connection_address.hostname(),
					 connection_address.port());
		});

		channel->onError([](char const *msg) { error("Channel error: {}", msg); });

		channel_created = true;
	}
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

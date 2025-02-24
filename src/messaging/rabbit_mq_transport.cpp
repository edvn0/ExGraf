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
			: work_guard_{boost::asio::make_work_guard(io_ctx_)}, handler_{io_ctx_},
				connection_{&handler_, AMQP::Address(conn)}, channel_{&connection_} {
		channel_.onReady([this, addr = AMQP::Address{conn}]() {
			channel_.declareQueue("metrics", AMQP::durable);
			channel_.declareExchange("metrics", AMQP::fanout, AMQP::durable);
			channel_.bindQueue("metrics", "metrics", "metrics");
			is_connected_ = true;

			info("Connected to RabbitMQ at {}:{}", addr.hostname(), addr.port());
		});
		channel_.onError([](char const *msg) { error("Channel error: {}", msg); });
		runner_ = std::jthread([this]() { io_ctx_.run(); });
	}

	~RabbitMQTransportImpl() {
		try {
			close_transport();
		} catch (std::exception const &e) {
			error("Exception in ~RabbitMQTransportImpl: {}", e.what());
		}
	}

	static constexpr auto to_rabbitmq_exchange = [](const UI::Outbox outbox) {
		switch (outbox) {
		case UI::Outbox::Metrics:
			return "metrics";
		case UI::Outbox::ModelConfiguration:
			return "model_configuration";
		}
		return "unknown";
	};

	auto send_impl(const UI::MessageTo &message) -> void {
		if (is_connected_) {
			AMQP::Envelope envelope(message.message.data(), message.message.size());
			auto exchange = to_rabbitmq_exchange(message.outbox);
			auto routing_key = exchange;
			channel_.publish(exchange, routing_key, envelope);
		}
	}

	auto close_transport() -> void {
		if (!closed_) {
			if (is_connected_) {
				channel_.close();
				connection_.close();
				boost::asio::steady_timer timer(io_ctx_);
				timer.expires_after(std::chrono::milliseconds(200));
				timer.async_wait([](auto) {});
				io_ctx_.run_for(std::chrono::milliseconds(250));
			}
			work_guard_.reset();
			io_ctx_.stop();
			if (runner_.joinable())
				runner_.join();
			closed_ = true;
		}
	}

	auto wait_for_connection() -> void {
		while (!is_connected_) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

private:
	boost::asio::io_context io_ctx_;
	boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
			work_guard_;
	AMQP::LibBoostAsioHandler handler_;
	AMQP::TcpConnection connection_;
	AMQP::TcpChannel channel_;
	bool is_connected_{false};
	bool closed_{false};
	std::jthread runner_;
};

RabbitMQTransport::RabbitMQTransport(std::string const &conn_string)
		: impl(std::make_unique<RabbitMQTransportImpl>(conn_string)) {}

RabbitMQTransport::~RabbitMQTransport() = default;

auto RabbitMQTransport::send_impl(const UI::MessageTo &message) -> void {
	impl->send_impl(message);
}

auto RabbitMQTransport::shutdown() -> void { impl->close_transport(); }

auto RabbitMQTransport::wait_for_connection() -> void {
	impl->wait_for_connection();
}

} // namespace ExGraf::Messaging

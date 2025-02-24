#include "exgraf/messaging/rabbit_mq_transport.hpp"
#include <amqpcpp.h>
#include <amqpcpp/linux_tcp.h>
#include <boost/asio.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace ExGraf::Messaging {

class RabbitMQTransport::RabbitMQTransportImpl {
public:
	explicit RabbitMQTransportImpl(const std::string &address)
			: io_context_(), work_(boost::asio::make_work_guard(io_context_)),
				handler_(io_context_), socket_(io_context_),
				connection_(&handler_, AMQP::Address(address)),
				channel_(std::make_unique<AMQP::Channel>(&connection_)) {
		channel_->declareQueue("metrics");
		worker_ = std::thread([this] { io_context_.run(); });
	}

	~RabbitMQTransportImpl() {
		work_.reset();
		io_context_.stop();
		if (worker_.joinable()) {
			worker_.join();
		}
	}

	void send(const std::string_view message) {
		channel_->publish("", "metrics", message.data(), message.size());
	}

private:
	boost::asio::io_context io_context_;
	boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
			work_;
	AMQP::LibBoostAsioHandler handler_;
	boost::asio::ip::tcp::socket socket_;
	AMQP::TcpConnection connection_;
	std::unique_ptr<AMQP::Channel> channel_;
	std::thread worker_;
};

RabbitMQTransport::RabbitMQTransport(const std::string &address)
		: impl_(std::make_unique<RabbitMQTransportImpl>(address)) {}

RabbitMQTransport::~RabbitMQTransport() = default;

void RabbitMQTransport::send_impl(const std::string_view message) {
	impl_->send(message);
}

} // namespace ExGraf::Messaging

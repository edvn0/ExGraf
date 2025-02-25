#pragma once

#include "exgraf/messaging/metrics_logger_base.hpp"

namespace ExGraf::Messaging {

// Transport API concept
template <typename Transport>
concept TransportLike =
		requires(Transport transport, const MessageTo &log_entry) {
			{ transport.send(log_entry) } -> std::same_as<void>;
			{ transport.wait_for_connection() } -> std::same_as<void>;
			{ transport.shutdown() } -> std::same_as<void>;
		};

template <TransportLike Transport>
class BusMetricsLogger : public MetricsLoggerBase<BusMetricsLogger<Transport>> {
public:
	explicit BusMetricsLogger(const std::string &address) : transport(address) {}

	void write_log_impl(const MessageTo &log_entry) { transport.send(log_entry); }
	auto wait_for_connection_impl() -> void { transport.wait_for_connection(); }
	auto wait_for_shutdown_impl() -> void { transport.shutdown(); }

private:
	Transport transport;
};

} // namespace ExGraf::Messaging

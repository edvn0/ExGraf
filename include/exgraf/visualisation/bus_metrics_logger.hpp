#pragma once

#include "exgraf/visualisation/metrics_logger_base.hpp"

namespace ExGraf::UI {

template <typename Transport>
class BusMetricsLogger : public MetricsLoggerBase<BusMetricsLogger<Transport>> {
public:
	explicit BusMetricsLogger(const std::string &address) : transport(address) {}

	void write_log(const MessageTo &log_entry) { transport.send(log_entry); }

	auto wait_for_connection_impl() -> void { transport.wait_for_connection(); }
	auto wait_for_shutdown_impl() -> void { transport.shutdown(); }

private:
	Transport transport;
};

} // namespace ExGraf::UI

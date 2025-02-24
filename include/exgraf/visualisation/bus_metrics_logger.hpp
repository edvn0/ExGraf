#pragma once

#include "metrics_logger_base.hpp"

namespace ExGraf::UI {

template <typename Transport>
class BusMetricsLogger : public MetricsLoggerBase<BusMetricsLogger<Transport>> {
public:
	explicit BusMetricsLogger(const std::string &address) : transport(address) {}

	void write_log(const std::string_view log_entry) {
		transport.send(log_entry);
	}

private:
	Transport transport;
};

} // namespace ExGraf::UI

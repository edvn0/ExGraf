#pragma once

#include "exgraf/messaging/metrics_logger_base.hpp"

#include <fstream>

namespace ExGraf::Messaging {

struct FileNotOpenError : std::runtime_error {
	using std::runtime_error::runtime_error;
};

class FileMetricsLogger : public MetricsLoggerBase<FileMetricsLogger> {
public:
	explicit FileMetricsLogger(const std::string &filename)
			: stream(filename, std::ios::out | std::ios::app) {
		if (!stream.is_open()) {
			throw FileNotOpenError(fmt::format("Could not open file: {}", filename));
		}

		stream << "epoch,loss,accuracy,ppv,fpr,recall\n";
		stream.flush();
	}

	void write_log_impl(const MessageTo &log_entry) {
		stream << log_entry.message;
		stream.flush();
	}

	auto wait_for_connection_impl() -> void { return; }
	auto wait_for_shutdown_impl() -> void { stream.close(); }

private:
	std::ofstream stream;
};

} // namespace ExGraf::Messaging

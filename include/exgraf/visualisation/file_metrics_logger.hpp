#pragma once

#include "exgraf/visualisation/metrics_logger_base.hpp"
#include <fstream>

namespace ExGraf::UI {

class FileMetricsLogger : public MetricsLoggerBase<FileMetricsLogger> {
public:
	explicit FileMetricsLogger(const std::string &filename)
			: stream(filename, std::ios::out | std::ios::app) {
		if (!stream.is_open()) {
			throw std::runtime_error("Failed to open file: " + filename);
		}

		stream << "epoch,loss,accuracy,ppv,fpr,recall\n";
		stream.flush();
	}

	void write_log(const MessageTo &log_entry) {
		stream << log_entry.message;
		stream.flush();
	}

private:
	std::ofstream stream;
};

} // namespace ExGraf::UI

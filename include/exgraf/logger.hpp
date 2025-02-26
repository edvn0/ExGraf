#pragma once

#include <cstdlib>
#include <fmt/format.h>
#include <memory>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace ExGraf {

auto get_from_environment(std::string_view) -> std::string;

class Logger {
public:
	static auto instance() -> auto & {
		static std::shared_ptr<spdlog::logger> logger = [] {
			auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
			auto log = std::make_shared<spdlog::logger>("app_logger", sink);
			spdlog::register_logger(log);
			log->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
			log->set_level(spdlog::level::debug);
			if (const auto log_level = get_from_environment("LOG_LEVEL");
					log_level.empty()) {
				log->set_level(spdlog::level::from_str(log_level));
			}

			return log;
		}();
		return *logger;
	}

	static auto graphviz_instance() -> auto & {
		static std::shared_ptr<spdlog::logger> graphviz_logger = [] {
			auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
					"graphviz.dot", false);
			auto log = std::make_shared<spdlog::logger>("graphviz_logger", file_sink);
			spdlog::register_logger(log);
			if (const auto log_level = get_from_environment("LOG_LEVEL");
					log_level.empty()) {
				log->set_level(spdlog::level::from_str(log_level));
			} else {
				log->set_level(spdlog::level::debug);
			}

			return log;
		}();
		return *graphviz_logger;
	}
};

template <typename... Args>
static auto info(const fmt::format_string<Args...> &fmt, Args &&...args)
		-> void {
	Logger::instance().info("[INFO] {}",
													fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
static auto error(const fmt::format_string<Args...> &fmt, Args &&...args)
		-> void {
	Logger::instance().error("[ERROR] {}",
													 fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
static auto debug(const fmt::format_string<Args...> &fmt, Args &&...args)
		-> void {
	Logger::instance().debug("[DEBUG] {}",
													 fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
static auto trace(const fmt::format_string<Args...> &fmt, Args &&...args)
		-> void {
	Logger::instance().trace("[TRACE] {}",
													 fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
static auto warn(const fmt::format_string<Args...> &fmt, Args &&...args)
		-> void {
	Logger::instance().warn("[WARN] {}",
													fmt::format(fmt, std::forward<Args>(args)...));
}

inline auto log_graphviz(const std::string &graphviz_content) -> void {
	Logger::graphviz_instance().info(graphviz_content);
}

} // namespace ExGraf

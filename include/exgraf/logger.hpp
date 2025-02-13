#pragma once

#include <fmt/format.h>
#include <memory>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace ExGraf {

class Logger {
public:
  static auto instance() -> auto & {
    static std::shared_ptr<spdlog::logger> logger = [] {
      auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      auto log = std::make_shared<spdlog::logger>("app_logger", sink);
      spdlog::register_logger(log);
      log->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
      log->set_level(spdlog::level::debug);
      return log;
    }();
    return *logger;
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

} // namespace ExGraf
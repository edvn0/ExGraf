#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"

#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace ExGraf::Messaging {

template <typename T>
concept JsonSerializable = requires(T t, std::ostream out) {
	{ T::to_json(t) } -> std::same_as<std::string>;
	{ T::from_json(std::string{}) } -> std::same_as<T>;
	{ out << t };
};

enum class Outbox : std::uint8_t { Metrics, ModelConfiguration };

struct MessageTo {
	Outbox outbox;
	std::string message;
};

template <typename Derived> class MetricsLoggerBase {
public:
	explicit MetricsLoggerBase() {
		writer_thread = std::jthread(&MetricsLoggerBase::writer_loop, this);
	}

	~MetricsLoggerBase() {
		{
			std::lock_guard lock(mutex);
			done = true;
		}
		cv.notify_one();
		if (writer_thread.joinable()) {
			writer_thread.join();
		}
	}

	template <JsonSerializable T, Outbox O> auto write_object(T &&object) {
		std::ostringstream ss;
		ss << std::forward<T>(object) << "\n";
		{
			std::lock_guard lock(mutex);
			log_queue.emplace(O, ss.str());
		}
		cv.notify_one();
	}
	template <JsonSerializable T, Outbox O> auto write_object(const T &object) {
		std::ostringstream ss;
		ss << object << "\n";
		{
			std::lock_guard lock(mutex);
			log_queue.emplace(O, ss.str());
		}
		cv.notify_one();
	}

	auto wait_for_shutdown() -> void {
		if (writer_thread.joinable()) {
			writer_thread.join();
		}
		static_cast<Derived &>(*this).wait_for_shutdown_impl();
	}

	auto wait_for_connection() -> void {
		static_cast<Derived &>(*this).wait_for_connection_impl();
	}

private:
	bool done{false};

	std::queue<MessageTo> log_queue;
	std::mutex mutex;
	std::condition_variable cv;
	std::jthread writer_thread;

	auto writer_loop() {
		while (true) {
			std::unique_lock lock(mutex);
			cv.wait(lock, [this] { return !log_queue.empty() || done; });

			while (!log_queue.empty()) {
				static_cast<Derived &>(*this).write_log_impl(log_queue.front());
				log_queue.pop();
			}

			if (done) {
				break;
			}
		}
	}
};

} // namespace ExGraf::Messaging

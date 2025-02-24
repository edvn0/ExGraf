#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace ExGraf::UI {

template <typename T>
concept JsonSerializable = requires(T t, std::ostream out) {
	{ T::to_json(t) } -> std::same_as<std::string>;
	{ T::from_json(std::string{}) } -> std::same_as<T>;
	{ out << t };
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

	template <AllowedTypes T>
	void log(std::size_t epoch, T loss, T accuracy, T ppv, T fpr, T recall) {
		std::ostringstream ss;

		if constexpr (std::is_same_v<T, double>) {
			ss.precision(std::streamsize{8});
		}

		ss << epoch << "," << loss << "," << accuracy << "," << ppv << "," << fpr
			 << "," << recall << "\n";

		{
			std::lock_guard lock(mutex);
			log_queue.push(ss.str());
		}
		cv.notify_one();
	}

	template <JsonSerializable T> void write_object(T &&object) {
		std::ostringstream ss;
		ss << std::forward<T>(object) << "\n";
		{
			std::lock_guard lock(mutex);
			log_queue.push(ss.str());
		}
		cv.notify_one();
	}

	auto wait_for_shutdown() -> void {
		static_cast<Derived &>(*this).wait_for_shutdown_impl();
		if (writer_thread.joinable()) {
			writer_thread.join();
		}
	}

private:
	bool done{false};
	std::queue<std::string> log_queue;
	std::mutex mutex;
	std::condition_variable cv;
	std::jthread writer_thread;

	void writer_loop() {
		while (true) {
			std::unique_lock lock(mutex);
			cv.wait(lock, [this] { return !log_queue.empty() || done; });

			while (!log_queue.empty()) {
				static_cast<Derived &>(*this).write_log(log_queue.front());
				log_queue.pop();
			}

			if (done)
				break;
		}
	}
};

} // namespace ExGraf::UI

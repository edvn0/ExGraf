#pragma once

#include "exgraf/allowed_types.hpp"
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace ExGraf::UI {

template <typename Derived> class MetricsLoggerBase {
public:
	explicit MetricsLoggerBase() : done(false) {
		writer_thread = std::thread(&MetricsLoggerBase::writer_loop, this);
	}

	~MetricsLoggerBase() {
		{
			std::lock_guard<std::mutex> lock(mutex);
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

private:
	bool done;
	std::queue<std::string> log_queue;
	std::mutex mutex;
	std::condition_variable cv;
	std::thread writer_thread;

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

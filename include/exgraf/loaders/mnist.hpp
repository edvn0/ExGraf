#pragma once

#include "exgraf/filesystem.hpp"
#include "exgraf/http/client.hpp"
#include <armadillo>
#include <exception>
#include <string>

#include <taskflow/taskflow.hpp>

namespace ExGraf::MNIST {

inline auto load(const std::string_view images, const std::string_view labels)
		-> std::pair<arma::Mat<double>, arma::Mat<double>> {

	static Http::HttpClient client;
	static tf::Executor executor;
	tf::Taskflow flow;
	std::exception_ptr ptr{nullptr};

	std::vector<std::uint8_t> img_data;
	std::vector<std::uint8_t> lbl_data;

	auto img_load_task = flow.emplace([&] {
		auto response = client.get(images);
		if (!response.success) {
			ptr = std::current_exception();
			return;
		}
		auto compressed_data =
				std::vector<std::uint8_t>(response.body.begin(), response.body.end());
		img_data = FS::decompress_gzip(compressed_data);
	});

	auto lbl_load_task = flow.emplace([&] {
		auto response = client.get(labels);
		if (!response.success) {
			ptr = std::current_exception();
			return;
		}
		auto compressed_data =
				std::vector<std::uint8_t>(response.body.begin(), response.body.end());
		lbl_data = FS::decompress_gzip(compressed_data);
	});

	std::pair<arma::Mat<double>, arma::Mat<double>> result;
	auto decode_task = flow.emplace([&] {
		FS::decode_images(img_data, std::get<0>(result));
		FS::decode_labels(lbl_data, std::get<1>(result));
	});

	auto transpose = flow.emplace([&] {
		std::get<0>(result) = std::get<0>(result).t();
		std::get<1>(result) = std::get<1>(result).t();
	});

	img_load_task.precede(decode_task);
	lbl_load_task.precede(decode_task);

	auto final_check = flow.emplace([&p = ptr] {
		if (p) {
			std::rethrow_exception(p);
		}
	});

	transpose.succeed(decode_task);
	final_check.succeed(transpose);

	executor.run(flow).wait();
	return result;
}

} // namespace ExGraf::MNIST

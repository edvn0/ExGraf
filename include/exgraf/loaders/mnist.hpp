#pragma once

#include "exgraf/http/client.hpp"
#include <armadillo>
#include <exception>
#include <string>

#include <taskflow/taskflow.hpp>

namespace ExGraf::MNIST {

namespace detail {
auto parse_int(const std::vector<std::uint8_t> &, std::size_t) -> std::int32_t;
auto decode_images(const std::vector<std::uint8_t> &) -> arma::Mat<double>;
auto decode_labels(const std::vector<std::uint8_t> &) -> arma::Mat<double>;
auto decompress_gzip(const std::vector<std::uint8_t> &)
		-> std::vector<std::uint8_t>;
} // namespace detail

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
		img_data = detail::decompress_gzip(compressed_data);
	});

	auto lbl_load_task = flow.emplace([&] {
		auto response = client.get(labels);
		if (!response.success) {
			ptr = std::current_exception();
			return;
		}
		auto compressed_data =
				std::vector<std::uint8_t>(response.body.begin(), response.body.end());
		lbl_data = detail::decompress_gzip(compressed_data);
	});

	std::pair<arma::Mat<double>, arma::Mat<double>> result;
	auto decode_task = flow.emplace([&] {
		result = std::make_pair(detail::decode_images(img_data),
														detail::decode_labels(lbl_data));
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

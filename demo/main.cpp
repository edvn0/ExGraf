#include "exgraf/bus/models/model_configuration.hpp"
#include <exgraf.hpp>

#include <algorithm>
#include <armadillo>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace ExGraf;

template <typename T> using FourTuple = std::tuple<T, T, T, T>;
using BusConfigurationTuple = FourTuple<std::string>;

auto read_program_options(const std::int32_t argc, char **argv)
		-> std::optional<BusConfigurationTuple> {
	namespace po = boost::program_options;

	po::options_description options_desc("Options");
	options_desc.add_options()("user",
														 po::value<std::string>()->default_value("guest"),
														 "RabbitMQ username")(
			"password", po::value<std::string>()->default_value("guest"),
			"RabbitMQ password")("host",
													 po::value<std::string>()->default_value("localhost"),
													 "RabbitMQ host")(
			"port", po::value<std::string>()->default_value("5672"), "RabbitMQ port");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, options_desc), vm);
	po::notify(vm);

	if (!vm.count("user") || !vm.count("password") || !vm.count("host") ||
			!vm.count("port")) {
		fmt::print("Usage: {} --user <username> --password <password> --host "
							 "<host> --port <port>\n",
							 argv[0]);
		return std::nullopt;
	}

	std::string user_value = vm["user"].as<std::string>();
	std::string password_value = vm["password"].as<std::string>();
	std::string host_value = vm["host"].as<std::string>();
	std::string port_value = vm["port"].as<std::string>();

	return std::make_tuple(std::move(user_value), std::move(password_value),
												 std::move(host_value), std::move(port_value));
}

template <AllowedTypes T>
auto batch_predict(ExpressionGraph<T> &graph, const arma::Mat<double> &images,
									 const arma::Mat<double> &labels, std::size_t num_samples,
									 std::size_t batch_size,
									 const std::vector<std::size_t> &indices,
									 arma::Mat<std::size_t> &confusion_matrix)
		-> std::tuple<T, std::size_t> {
	T epoch_loss = 0.0;
	std::size_t correct_predictions = 0;
	for (std::size_t batch_start = 0; batch_start < num_samples;
			 batch_start += batch_size) {
		std::size_t current_batch_size =
				std::min(batch_size, num_samples - batch_start);
		arma::mat batch_images(current_batch_size, images.n_cols);
		arma::mat batch_labels(current_batch_size, labels.n_cols);
		for (std::size_t i = 0; i < current_batch_size; ++i) {
			std::size_t idx = indices[batch_start + i];
			batch_images.row(i) = images.row(idx);
			batch_labels.row(i) = labels.row(idx);
		}
		auto output = graph.predict(batch_images);
		T batch_loss = graph.train(batch_labels)(0, 0);
		epoch_loss += batch_loss * static_cast<T>(current_batch_size);
		for (std::size_t i = 0; i < current_batch_size; ++i) {
			arma::uword pred_label{};
			arma::uword true_label{};
			output.row(i).max(pred_label);
			batch_labels.row(i).max(true_label);
			confusion_matrix(true_label, pred_label)++;
			correct_predictions += static_cast<std::size_t>(pred_label == true_label);
		}
	}
	return {
			epoch_loss / static_cast<T>(num_samples),
			correct_predictions,
	};
}

struct Metrics {
	std::vector<double> positive_predictive_values;
	std::vector<double> false_positive_rates;
	std::vector<double> recalls;
};

auto compute_metrics(const arma::Mat<std::size_t> &confusion) -> Metrics {
	std::size_t n = confusion.n_rows;
	std::vector<double> positive_predictive_values(n, 0.0);
	std::vector<double> false_positive_rates(n, 0.0);
	std::vector<double> recalls(n, 0.0);
	std::size_t total = arma::accu(confusion);
	for (std::size_t i = 0; i < n; ++i) {
		std::size_t tp = confusion(i, i);
		std::size_t fp = 0;
		std::size_t fn = 0;
		for (std::size_t r = 0; r < n; ++r) {
			if (r != i)
				fp += confusion(r, i);
		}
		for (std::size_t c = 0; c < n; ++c) {
			if (c != i)
				fn += confusion(i, c);
		}

		static constexpr auto close_to_zero = [](std::floating_point auto const x) {
			return std::abs(x) < 1e-6;
		};

		std::size_t tn = total - (tp + fp + fn);
		auto true_predictive = static_cast<double>(tp);
		auto false_predictive = static_cast<double>(fp);
		auto total_positive = static_cast<double>(tp + fn);
		auto total_true = static_cast<double>(tp + fp);
		auto total_false = static_cast<double>(fn + tn);
		// What would fp + tn be called?
		auto total_negative = static_cast<double>(fp + tn);

		positive_predictive_values[i] =
				close_to_zero(total_true) ? 0.0 : true_predictive / total_true;
		recalls[i] =
				close_to_zero(total_positive) ? 0.0 : true_predictive / total_positive;
		false_positive_rates[i] =
				close_to_zero(total_negative) ? 0.0 : false_predictive / total_false;
	}
	return {
			std::move(positive_predictive_values),
			std::move(false_positive_rates),
			std::move(recalls),
	};
}

int main(int argc, char **argv) {
	auto bus_configuration = read_program_options(argc, argv);
	if (!bus_configuration) {
		return 1;
	}
	auto &&[user_value, password_value, host_value, port_value] =
			*bus_configuration;

	using T = double;
	ExpressionGraph<T> graph({784, 10, 10});
	graph.compile_model<ADAMOptimizer<T>>(
			{.input_size = {ExpressionGraph<T>::ModelConfig::unused, 784}}, 0.001,
			0.9, 0.99);

	auto &&[images, labels] =
			MNIST::load("https://raw.githubusercontent.com/fgnt/mnist/master/"
									"train-images-idx3-ubyte.gz",
									"https://raw.githubusercontent.com/fgnt/mnist/master/"
									"train-labels-idx1-ubyte.gz");

	std::size_t num_samples = images.n_rows;
	std::size_t num_epochs = 300;
	std::size_t batch_size = 128;
	std::vector<std::size_t> indices(num_samples);
	std::ranges::iota(indices, 0);
	std::random_device rd;
	std::mt19937 g(rd());

	graph.visit<GraphvizVisitor<T>>("mnist_other.dot",
																	VisualisationMode::LeftToRight);

#ifdef USE_ZERO_MQ
	Messaging::BusMetricsLogger<Messaging::ZeroMQTransport> logger(
			"tcp://*:5555");
#else
	std::string amqp_uri = fmt::format("amqp://{}:{}@{}:{}/", user_value,
																		 password_value, host_value, port_value);
	Messaging::BusMetricsLogger<Messaging::RabbitMQTransport> logger(amqp_uri);
	logger.wait_for_connection();
#endif

	Bus::Models::ModelConfiguration model = {
			.name = "MNIST",
			.layers = std::vector<std::size_t>{784, 10, 10},
			.optimizer = "ADAM",
			.learning_rate = 0.001,
	};
	logger.write_object<Bus::Models::ModelConfiguration,
											Messaging::Outbox::ModelConfiguration>(model);

	for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
		arma::Mat<size_t> confusion_matrix(10, 10, arma::fill::zeros);
		std::ranges::shuffle(indices, g);
		auto &&[epoch_loss, correct_predictions] =
				batch_predict(graph, images, labels, num_samples, batch_size, indices,
											confusion_matrix);
		T accuracy =
				static_cast<T>(correct_predictions) / static_cast<T>(num_samples);
		auto &&[positive_predictive_values, false_positive_rates, recalls] =
				compute_metrics(confusion_matrix);
		auto mean_ppv = arma::mean(arma::vec(positive_predictive_values));
		auto mean_fpr = arma::mean(arma::vec(false_positive_rates));
		auto mean_recall = arma::mean(arma::vec(recalls));
		logger
				.write_object<Bus::Models::MetricsMessage, Messaging::Outbox::Metrics>({
						.epoch = static_cast<std::int32_t>(epoch + 1),
						.loss = epoch_loss,
						.accuracy = accuracy,
						.mean_ppv = mean_ppv,
						.mean_fpr = mean_fpr,
						.mean_recall = mean_recall,
						.model_configuration = &model,
				});
		info("Epoch {}/{}: Loss={:.4f}, Accuracy={:.2f}%, PPV={:.4f}, "
				 "FPR={:.4f}, Recall={:.4f}\n",
				 epoch + 1, num_epochs, epoch_loss, accuracy * 100.0, mean_ppv,
				 mean_fpr, mean_recall);
	}
	logger.wait_for_shutdown();
	return 0;
}

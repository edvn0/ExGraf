#include <exgraf.hpp>

#include "exgraf/messaging/rabbit_mq_transport.hpp"
#include "exgraf/messaging/zero_mq_transport.hpp"
#include "exgraf/visualisation/bus_metrics_logger.hpp"

#include <algorithm>
#include <armadillo>

#include <fmt/format.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace ExGraf;

template <AllowedTypes T>
auto batch_predict(ExpressionGraph<T> &graph, const arma::Mat<double> &images,
									 const arma::Mat<double> &labels, size_t num_samples,
									 size_t batch_size, const std::vector<size_t> &indices,
									 arma::Mat<size_t> &confusion_matrix)
		-> std::tuple<T, size_t> {
	T epoch_loss = 0.0;
	size_t correct_predictions = 0;
	for (size_t batch_start = 0; batch_start < num_samples;
			 batch_start += batch_size) {
		size_t current_batch_size = std::min(batch_size, num_samples - batch_start);
		arma::mat batch_images(current_batch_size, images.n_cols);
		arma::mat batch_labels(current_batch_size, labels.n_cols);
		for (size_t i = 0; i < current_batch_size; ++i) {
			size_t idx = indices[batch_start + i];
			batch_images.row(i) = images.row(idx);
			batch_labels.row(i) = labels.row(idx);
		}
		auto output = graph.predict(batch_images);
		T batch_loss = graph.train(batch_labels)(0, 0);
		epoch_loss += batch_loss * static_cast<T>(current_batch_size);
		for (size_t i = 0; i < current_batch_size; ++i) {
			arma::uword pred_label{};
			arma::uword true_label{};
			output.row(i).max(pred_label);
			batch_labels.row(i).max(true_label);
			confusion_matrix(true_label, pred_label)++;
			correct_predictions += static_cast<size_t>(pred_label == true_label);
		}
	}
	return {epoch_loss / static_cast<T>(num_samples), correct_predictions};
}

struct Metrics {
	std::vector<double> ppvs;
	std::vector<double> fprs;
	std::vector<double> recalls;
};

auto compute_metrics(const arma::Mat<size_t> &confusion) -> Metrics {
	size_t n = confusion.n_rows;
	std::vector<double> ppvs(n, 0.0);
	std::vector<double> fprs(n, 0.0);
	std::vector<double> recalls(n, 0.0);
	size_t total = arma::accu(confusion);
	for (size_t i = 0; i < n; ++i) {
		size_t tp = confusion(i, i);
		size_t fp = 0;
		size_t fn = 0;
		for (size_t r = 0; r < n; ++r) {
			if (r != i)
				fp += confusion(r, i);
		}
		for (size_t c = 0; c < n; ++c) {
			if (c != i)
				fn += confusion(i, c);
		}
		size_t tn = total - (tp + fp + fn);
		ppvs[i] = (tp + fp == 0) ? 0.0 : double(tp) / double(tp + fp);
		recalls[i] = (tp + fn == 0) ? 0.0 : double(tp) / double(tp + fn);
		fprs[i] = (fp + tn == 0) ? 0.0 : double(fp) / double(fp + tn);
	}
	return {
			std::move(ppvs),
			std::move(fprs),
			std::move(recalls),
	};
}

int main(int, char **, char **) {
	using T = double;
	ExpressionGraph<T> graph({784, 10, 10});
	graph.compile_model<ADAMOptimizer<T>>(
			{
					.input_size =
							{
									ExpressionGraph<T>::ModelConfig::unused,
									784,
							},
			},
			0.001, 0.9, 0.99);

	auto &&[images, labels] =
			MNIST::load("https://raw.githubusercontent.com/fgnt/mnist/master/"
									"train-images-idx3-ubyte.gz",
									"https://raw.githubusercontent.com/fgnt/mnist/master/"
									"train-labels-idx1-ubyte.gz");

	size_t num_samples = images.n_rows;
	size_t num_epochs = 300;
	size_t batch_size = 128;
	std::vector<size_t> indices(num_samples);
	std::ranges::iota(indices, 0);
	std::random_device rd;
	std::mt19937 g(rd());

	graph.visit<GraphvizVisitor<T>>("mnist_other.dot",
																	VisualisationMode::LeftToRight);

#ifdef USE_ZERO_MQ
	UI::BusMetricsLogger<Messaging::ZeroMQTransport> logger("tcp://*:5555");
#else
	UI::BusMetricsLogger<Messaging::RabbitMQTransport> logger(
			"amqp://guest:guest@localhost/");
#endif

	for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
		arma::Mat<size_t> confusion_matrix(10, 10, arma::fill::zeros);
		std::ranges::shuffle(indices, g);
		auto &&[epoch_loss, correct_predictions] =
				batch_predict(graph, images, labels, num_samples, batch_size, indices,
											confusion_matrix);

		T accuracy =
				static_cast<T>(correct_predictions) / static_cast<T>(num_samples);
		auto &&[ppvs, fprs, recalls] = compute_metrics(confusion_matrix);
		const auto mean_ppv = arma::mean(arma::vec(ppvs));
		const auto mean_fpr = arma::mean(arma::vec(fprs));
		const auto mean_recall = arma::mean(arma::vec(recalls));

		logger.log(epoch + 1, epoch_loss, accuracy * 100.0, mean_ppv, mean_fpr,
							 mean_recall);

		fmt::print("Epoch {}/{}: Loss={:.4f}, Accuracy={:.2f}%, PPV={:.4f}, "
							 "FPR={:.4f}, Recall={:.4f}\n",
							 epoch + 1, num_epochs, epoch_loss, accuracy * 100.0, mean_ppv,
							 mean_fpr, mean_recall);
	}

	return 0;
}

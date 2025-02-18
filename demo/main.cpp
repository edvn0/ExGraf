#include "exgraf/expression_graph.hpp"
#include "exgraf/visitors/evaluation_order.hpp"
#include <algorithm>
#include <armadillo>
#include <exgraf.hpp>
#include <fmt/format.h>
#include <random>
#include <vector>

using namespace ExGraf;

int main() {
	using T = double;

	ExpressionGraph<T> graph({784, 256, 10});
	graph.compile_model({
			.input_size =
					{
							ExpressionGraph<T>::ModelConfig::unused,
							784,
					},
	});

	auto &&[images, labels] =
			MNIST::load("https://raw.githubusercontent.com/fgnt/mnist/master/"
									"t10k-images-idx3-ubyte.gz",
									"https://raw.githubusercontent.com/fgnt/mnist/master/"
									"t10k-labels-idx1-ubyte.gz");

	size_t num_samples = images.n_rows;

	const size_t num_epochs = 5;
	const size_t batch_size = 32;

	std::vector<size_t> indices(num_samples);
	std::iota(indices.begin(), indices.end(), 0);

	std::random_device rd;
	std::mt19937 g(rd());

	graph.visit<GraphvizVisitor<T>>("mnist_other.dot",
																	VisualisationMode::LeftToRight);
	graph.visit<GraphEvaluationVisitor<T>>();

	for (std::size_t epoch = 0; epoch < num_epochs; ++epoch) {
		std::shuffle(indices.begin(), indices.end(), g);

		T epoch_loss = 0.0;
		std::size_t correct_predictions = 0;

		for (std::size_t batch_start = 0; batch_start < num_samples;
				 batch_start += batch_size) {
			size_t current_batch_size =
					std::min(batch_size, num_samples - batch_start);

			arma::mat batch_images(current_batch_size, images.n_cols);
			arma::mat batch_labels(current_batch_size, labels.n_cols);

			for (std::size_t i = 0; i < current_batch_size; ++i) {
				std::size_t idx = indices[batch_start + i];
				batch_images.row(i) = images.row(idx);
				batch_labels.row(i) = labels.row(idx);
			}

			auto output = graph.predict(batch_images);
			T batch_loss = graph.learn(batch_labels)(0, 0);
			epoch_loss += batch_loss * static_cast<T>(current_batch_size);

			for (std::size_t i = 0; i < current_batch_size; ++i) {
				arma::uword pred_label, true_label;
				output.row(i).max(pred_label);
				batch_labels.row(i).max(true_label);
				if (pred_label == true_label) {
					correct_predictions++;
				}
			}
		}

		epoch_loss /= static_cast<T>(num_samples);
		const auto accuracy =
				static_cast<T>(correct_predictions) / static_cast<T>(num_samples);

		fmt::print("Epoch {}/{}: Loss = {:.4f}, Accuracy = {:.2f}%\n", epoch + 1,
							 num_epochs, epoch_loss, accuracy * 100.0);
	}

	return 0;
}

#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/node.hpp"
#include "exgraf/operations.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/variable.hpp"

#include <armadillo>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace ExGraf {

enum class ActivationFunction : std::uint8_t {
	ReLU,
};
enum class LossFunction : std::uint8_t {
	MeanSquaredError,
	CrossEntropy,
};
enum class OutputActivationFunction : std::uint8_t {
	Softmax,
	ReLU,
};
enum class Optimizer : std::uint8_t {
	SGD,
};

template <AllowedTypes T>
static constexpr auto randn_matrix(std::uint32_t rows, std::uint32_t cols) {
	arma::Mat<T> matrix(cols, rows, arma::fill::randn);
	return matrix;
}

template <AllowedTypes T>
arma::Mat<T> initialize_weights(std::uint32_t input_size,
																std::uint32_t output_size,
																ActivationFunction activation) {
	if (activation == ActivationFunction::ReLU) {
		return randn_matrix<T>(input_size, output_size) *
					 std::sqrt(2.0 / input_size);
	} else {
		return randn_matrix<T>(input_size, output_size) *
					 std::sqrt(2.0 / (input_size + output_size));
	}
}

template <AllowedTypes T> class ExpressionGraph {
public:
	ExpressionGraph(const std::initializer_list<std::uint32_t> &sizes)
			: layer_sizes(sizes) {}

	struct ModelConfig {
		ActivationFunction activation_function{ActivationFunction::ReLU};
		LossFunction loss_function{LossFunction::CrossEntropy};
		OutputActivationFunction output_activation_function{
				OutputActivationFunction::Softmax};
		Optimizer optimizer{Optimizer::SGD};
	};
	void compile_model(const ModelConfig &config) {
		auto x = add_placeholder("X");
		auto y = add_placeholder("Y");
		input = x;

		// First layer
		// If input X is (n × d_0), then W_0 should be (d_0 × d_1)
		auto w0 = add_variable(randn_matrix<T>(layer_sizes[0], layer_sizes[1]));
		auto b0 = add_variable(randn_matrix<T>(1, layer_sizes[1]));

		// z0 = X · W_0, resulting in (n × d_1)
		auto z0 = add_node<Mult<T>>(x, w0);
		auto z1 = add_node<Add<T>>(z0, b0);

		auto layer_output = add_node<ReLU<T>>(z1);

		// Hidden layers
		for (std::size_t i = 1; i < layer_sizes.size() - 2; ++i) {
			// W_i should be (d_i × d_{i+1})
			auto w =
					add_variable(randn_matrix<T>(layer_sizes[i], layer_sizes[i + 1]));
			auto b = add_variable(randn_matrix<T>(1, layer_sizes[i + 1]));

			// If previous output is (n × d_i), then result will be (n × d_{i+1})
			auto z_hidden_0 = add_node<Mult<T>>(layer_output, w);
			auto z_hidden_1 = add_node<Add<T>>(z_hidden_0, b);
			layer_output = add_node<ReLU<T>>(z_hidden_1);
		}

		// Output layer - no activation yet
		auto w_last = add_variable(randn_matrix<T>(
				layer_sizes[layer_sizes.size() - 2], layer_sizes.back()));
		auto b_last = add_variable(randn_matrix<T>(1, layer_sizes.back()));
		auto mult_output = add_node<Mult<T>>(layer_output, w_last);
		auto add_output = add_node<Add<T>>(mult_output, b_last);

		// Apply output activation
		switch (config.output_activation_function) {
		case OutputActivationFunction::Softmax:
			predictor = add_node<Softmax<T>>(add_output);
			break;
		case OutputActivationFunction::ReLU:
			predictor = add_node<ReLU<T>>(add_output);
			break;
		default:
			throw std::runtime_error("Unsupported output activation function");
		}

		// Loss function
		switch (config.loss_function) {
		case LossFunction::CrossEntropy: {
			auto loss_output = add_node<CrossEntropyLoss<T>>(predictor, y);
			this->output = loss_output;
			break;
		}
		default:
			throw std::runtime_error("Unsupported loss function");
		}
	}

	arma::Mat<T> predict(const arma::Mat<T> &input_matrix) {
		if (input_matrix.n_cols != layer_sizes[0]) {
			throw std::logic_error(fmt::format(
					"Expected input shape (N, {}), but got ({} x {})", layer_sizes[0],
					input_matrix.n_rows, input_matrix.n_cols));
		}
		get_placeholder("X")->set_value(input_matrix.t());
		return predictor->forward();
	}

	void learn(const arma::Mat<T> &labels) {
		arma::Mat<T> loss_gradient = arma::ones<arma::Mat<T>>(10, 1);
		get_placeholder("Y")->set_value(labels);
		output->backward(loss_gradient);
	}

	template <typename Visitor, typename... Args> auto visit(Args &&...args) {
		auto visitor = std::make_unique<Visitor>(std::forward<Args>(args)...);
		if (output) {
			output->accept(*visitor);
		}

		if constexpr (requires(Visitor &t) { t.finalise(); }) {
			visitor->finalise();
		}
	}

	auto get_nodes() const {
		return nodes | std::views::transform([](auto &v) { return v.get(); });
	}

private:
	std::vector<std::uint32_t> layer_sizes;
	std::vector<std::unique_ptr<Node<T>>> nodes;
	std::unordered_map<std::string, Placeholder<T> *> placeholders;
	Placeholder<T> *input;
	Node<T> *output;
	Node<T> *predictor;

	template <typename NodeType, typename... Args>
	NodeType *add_node(Args &&...args) {
		auto node = std::make_unique<NodeType>(std::forward<Args>(args)...);
		NodeType *ptr = node.get();
		nodes.emplace_back(std::move(node));
		return ptr;
	}

	Placeholder<T> *add_placeholder(const std::string &id) {
		auto *placeholder = add_node<Placeholder<T>>(id);
		placeholders[id] = placeholder;
		return placeholder;
	}

	Variable<T> *add_variable(const arma::Mat<T> &initial_value) {
		return add_node<Variable<T>>(initial_value);
	}

	Placeholder<T> *get_placeholder(const std::string &id) {
		return placeholders.at(id);
	}
};

} // namespace ExGraf

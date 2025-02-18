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
		static constexpr auto unused = ~std::int32_t(0);
		static constexpr auto determine_samples = -1;
		arma::vec3 input_size{
				determine_samples,
				32,
				unused,
		};
		ActivationFunction activation_function{ActivationFunction::ReLU};
		LossFunction loss_function{LossFunction::CrossEntropy};
		OutputActivationFunction output_activation_function{
				OutputActivationFunction::Softmax};
		Optimizer optimizer{Optimizer::SGD};
	};
	auto compile_model(const ModelConfig &config) -> void {
		auto x = add_placeholder("X");
		auto y = add_placeholder("Y");
		input = x;

		Node<T> *current_layer = x;
		std::uint32_t current_size = layer_sizes[0];

		for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
			std::uint32_t output_size = layer_sizes[i + 1];
			auto layer = add_layer(current_layer, current_size, output_size);
			const auto is_last_layer = i < layer_sizes.size() - 2;
			if (!is_last_layer) {
				current_layer = add_node<ReLU<T>>(layer);
			} else {
				current_layer = layer;
			}
			current_size = output_size;
		}

		switch (config.output_activation_function) {
		case OutputActivationFunction::Softmax:
			predictor = add_node<Softmax<T>>(current_layer);
			break;
		case OutputActivationFunction::ReLU:
			predictor = add_node<ReLU<T>>(current_layer);
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
		get_placeholder("X")->set_value(input_matrix);
		return predictor->forward();
	}

	auto learn(const arma::Mat<T> &labels) -> arma::Mat<T> {
		arma::Mat<T> loss_gradient = arma::Mat<T>(10, 1, arma::fill::ones);
		get_placeholder("Y")->set_value(labels);
		output->backward(loss_gradient);
		return arma::Mat<T>{1, 1};
	}

	template <typename Visitor, typename... Args> auto visit(Args &&...args) {
		Visitor visitor{
				std::forward<Args>(args)...,
		};
		if (output) {
			output->accept(visitor);
		}

		if constexpr (requires(Visitor &t) { t.finalise(); }) {
			visitor.finalise();
		}

		if constexpr (requires(Visitor &t) { t.result(); }) {
			return visitor.result();
		}
	}

	auto get_nodes() const {
		return nodes | std::views::transform([](auto &v) { return v.get(); });
	}

	auto get_placeholder(const std::string &id) -> Placeholder<T> * {
		assert(placeholders.contains(id));
		return placeholders.at(id);
	}
	auto add_placeholder(const std::string &id) -> Placeholder<T> * {
		auto *placeholder = add_node<Placeholder<T>>(id);
		placeholders[id] = placeholder;
		return placeholder;
	}
	auto add_layer(Node<T> *input_node, std::unsigned_integral auto input_nodes,
								 std::unsigned_integral auto output_nodes) {
		assert(input_node);

		auto B = add_variable(randn_matrix<T>(output_nodes, 1));
		auto W = add_variable(randn_matrix<T>(output_nodes, input_nodes));
		auto XW = add_node<Mult<T>>(input_node, W);
		auto XW_plus_B = add_node<Add<T>>(XW, B);
		return XW_plus_B;
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

	Variable<T> *add_variable(const arma::Mat<T> &initial_value) {
		return add_node<Variable<T>>(initial_value);
	}
};

} // namespace ExGraf

#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/node.hpp"
#include "exgraf/operations.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/variable.hpp"

#include <armadillo>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ExGraf {

template <AllowedTypes T>
static constexpr auto randn_matrix(std::uint32_t rows, std::uint32_t cols) {
	arma::Mat<T> matrix(rows, cols, arma::fill::randn);
	return matrix;
}

template <AllowedTypes T> class ExpressionGraph {
public:
	std::vector<std::unique_ptr<Node<T>>> nodes;
	std::unordered_map<std::string, Placeholder<T> *> placeholders;
	Placeholder<T> *input;
	Node<T> *output;

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

	ExpressionGraph(const std::initializer_list<std::uint32_t> &sizes)
			: layer_sizes(sizes) {}

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

		auto w0 = add_variable(randn_matrix<T>(layer_sizes[0], layer_sizes[1]));
		auto b0 = add_variable(randn_matrix<T>(1, layer_sizes[1]));
		auto z0 = add_node<Mult<T>>(w0, x);
		auto z1 = add_node<Add<T>>(z0, b0);
		auto z2 = add_node<ReLU<T>>(z1);
		auto layer_output = z2;
		for (std::size_t i = 1; i < layer_sizes.size() - 2; ++i) {
			auto w =
					add_variable(randn_matrix<T>(layer_sizes[i], layer_sizes[i + 1]));
			auto b = add_variable(randn_matrix<T>(1, layer_sizes[i + 1]));
			auto z_hidden_0 = add_node<Mult<T>>(w, layer_output);
			auto z_hidden_1 = add_node<Add<T>>(z_hidden_0, b);
			auto z_hidden_2 = add_node<ReLU<T>>(z_hidden_1);
			layer_output = z_hidden_2;
		}

		// Last layer is softmax
		auto w = add_variable(randn_matrix<T>(layer_sizes[layer_sizes.size() - 2],
																					layer_sizes.back()));
		auto b = add_variable(randn_matrix<T>(1, layer_sizes.back()));
		auto mult_output = add_node<Mult<T>>(w, layer_output);
		auto add_output = add_node<Add<T>>(mult_output, b);
		auto output = add_node<Softmax<T>>(add_output);

		switch (config.loss_function) {
		case LossFunction::CrossEntropy: {
			auto loss_output = add_node<CrossEntropyLoss<T>>(output, y);
			this->output = loss_output;
			break;
		}
		default:
			throw std::runtime_error("Unsupported loss function");
		}
	}

	arma::Mat<T> predict(const arma::Mat<T> &input_matrix,
											 const std::string &placeholder_id) {
		get_placeholder(placeholder_id)->set_value(input_matrix);
		return output->compute();
	}

	void learn() {
		arma::Mat<T> loss_gradient =
				arma::ones<arma::Mat<T>>(output->value->n_rows, output->value->n_cols);
		output->backward(loss_gradient);
	}

private:
	std::vector<std::uint32_t> layer_sizes;
};

} // namespace ExGraf

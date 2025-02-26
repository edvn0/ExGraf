#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/model_configuration.hpp"
#include "exgraf/node.hpp"
#include "exgraf/node_visitor.hpp"
#include "exgraf/operations.hpp"
#include "exgraf/optimizers/adam_optimizer.hpp"
#include "exgraf/optimizers/optimizer.hpp"
#include "exgraf/optimizers/sgd_optimizer.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/variable.hpp"

#include <armadillo>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace ExGraf {

struct LayerConfig {
	std::uint32_t size;
	ActivationFunction activation;
	std::string name;
};

template <AllowedTypes T> class Sequential {
	using Var = Variable<T>;
	using Ph = Placeholder<T>;
	using Mat = arma::Mat<T>;
	using N = Node<T>;

public:
	Sequential() = default;

	void add_layer(std::uint32_t size, ActivationFunction activation,
								 const std::string &name = "") {
		layers.emplace_back(size, activation, name);
	}

	template <typename Opt, typename... OptimizerArgs>
	void compile(OptimizerArgs &&...args) {
		if (layers.empty()) {
			throw std::logic_error("Model must have at least one layer.");
		}

		optimizer = std::make_unique<Opt>(std::forward<OptimizerArgs>(args)...);
		auto x = add_placeholder("X");
		input = x;

		N *current_layer = x;
		std::uint32_t current_size = layers.front().size;

		for (size_t i = 0; i < layers.size(); ++i) {
			const auto &layer = layers[i];
			auto layer_output =
					add_dense_layer(current_layer, current_size, layer.size);
			auto *activation = create_activation(layer_output, layer.activation);
			current_layer = activation;
			current_size = layer.size;
		}
		output = current_layer;
	}

	Mat predict(const Mat &input_matrix) {
		if (input_matrix.n_cols != layers.front().size) {
			throw InvalidInputShapeError(
					"Input shape does not match model input size");
		}
		get_placeholder("X")->set_value(input_matrix);
		return output->forward();
	}

	void train(const Mat &labels) {
		get_placeholder("Y")->set_value(labels);
		Mat grad(1, 1, arma::fill::ones);
		(void)output->backward(grad);
		optimizer->step(std::span(trainable_nodes));
		traverse([](Node<T> &node) { node.zero_gradient(); });
	}

	template <typename F> void traverse(F &&f) {
		if (output) {
			auto visitor = make_visitor<T>(std::forward<F>(f));
			output->accept(visitor);
		}
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

private:
	std::vector<LayerConfig> layers;
	std::vector<std::unique_ptr<Node<T>>> nodes;
	std::vector<Var *> trainable_nodes;
	std::unordered_map<std::string, Ph *> placeholders;
	std::unique_ptr<Optimizer<T>> optimizer;
	Ph *input;
	N *output;

	Ph *add_placeholder(const std::string &id) {
		auto *placeholder = add_node<Placeholder<T>>(id);
		placeholders[id] = placeholder;
		return placeholder;
	}

	N *add_dense_layer(N *input_node, std::uint32_t input_nodes,
										 std::uint32_t output_nodes) {
		auto B = add_variable(randn_matrix<T>(output_nodes, 1));
		auto W = add_variable(randn_matrix<T>(output_nodes, input_nodes));
		auto XW = add_node<Mult<T>>(input_node, W);
		auto XW_plus_B = add_node<Add<T>>(XW, B);
		return XW_plus_B;
	}

	template <typename NodeType, typename... Args>
	NodeType *add_node(Args &&...args) {
		auto node = std::make_unique<NodeType>(std::forward<Args>(args)...);
		NodeType *ptr = node.get();
		nodes.emplace_back(std::move(node));
		return ptr;
	}

	auto add_variable(const arma::Mat<T> &initial_value) -> Var * {
		return add_node<Variable<T>>(initial_value);
	}

	auto get_placeholder(const std::string &id) -> Placeholder<T> * {
		return placeholders.at(id);
	}

	N *create_activation(Node<T> *input_node, ActivationFunction activation) {
		switch (activation) {
		case ActivationFunction::ReLU: {
			auto *relu = add_node<ReLU<T>>(input_node);
			return relu;
		}
		case ActivationFunction::Tanh: {
			auto *tanh = add_node<Tanh<T>>(input_node);
			return tanh;
		}
		default:
			throw UnsupportedOperationError("Unsupported activation function");
		}
	}
};

} // namespace ExGraf

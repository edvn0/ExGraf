#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/forward.hpp"
#include "exgraf/node_visitor.hpp"

#include <armadillo>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class Node {
protected:
	std::vector<Node<T> *> inputs;
	std::vector<Node<T> *> outputs;
	std::optional<arma::Mat<T>> value;
	arma::Mat<T> gradient;

	auto get_inputs() -> std::tuple<Node<T> *, Node<T> *> {
		return {
				inputs.at(0),
				inputs.at(1),
		};
	}

public:
	explicit Node(std::vector<Node<T> *> predecessors)
			: inputs(std::move(predecessors)) {
		for (auto *input : inputs) {
			input->outputs.push_back(this);
		}
	}

	explicit Node(std::initializer_list<Node<T> *> predecessors)
			: inputs(predecessors) {
		for (auto *input : inputs) {
			input->outputs.push_back(this);
		}
	}

	virtual ~Node() = default;
	virtual auto forward() -> arma::Mat<T> = 0;
	virtual auto backward(const arma::Mat<T> &) -> void = 0;
	virtual auto accept(NodeVisitor<T> &visitor) -> void = 0;
	virtual auto name() const -> std::string_view = 0;

	auto get_all_inputs() const { return std::span(inputs); }
	auto rows() const {
		if (!value)
			throw std::runtime_error("What");
		return value->n_rows;
	}
	auto cols() const {
		if (!value)
			throw std::runtime_error("What");
		return value->n_cols;
	}
	auto is_bias() const -> bool {
		if (value) {
			return value->n_rows == 1;
		}
		return false;
	}
	auto is_weight() const -> bool {
		if (value) {
			return value->n_rows != 1;
		}
		return false;
	}
	auto has_value() const { return value.has_value(); }

	friend class ExpressionGraph<T>;
};

} // namespace ExGraf

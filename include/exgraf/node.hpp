#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/forward.hpp"
#include "exgraf/node_visitor.hpp"

#include <armadillo>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ExGraf {

struct MissingValueError : public std::logic_error {
	using std::logic_error::logic_error;
};

enum class NodeType : std::uint8_t {
	Variable,
	Placeholder,
	Add,
	Mult,
	Softmax,
	CrossEntropyLoss,
	ReLU,
	Sum,
	Negate,
	Log,
	Hadamard,
};

template <AllowedTypes T> class Node {
protected:
	NodeType type;
	std::vector<Node<T> *> inputs;
	std::vector<Node<T> *> outputs;
	std::optional<arma::Mat<T>> value{arma::Mat<T>(0, 0)};
	arma::Mat<T> gradient;

	auto get_inputs() -> std::tuple<Node<T> *, Node<T> *> {
		return {
				inputs.at(0),
				inputs.at(1),
		};
	}

	explicit Node(NodeType t, std::vector<Node<T> *> predecessors)
			: type(t), inputs(std::move(predecessors)) {
		for (auto *input : inputs) {
			input->outputs.push_back(this);
		}
	}

	explicit Node(NodeType t, std::initializer_list<Node<T> *> predecessors)
			: type(t), inputs(predecessors) {
		for (auto *input : inputs) {
			input->outputs.push_back(this);
		}
	}

public:
	virtual ~Node() = default;
	virtual auto forward() -> arma::Mat<T> = 0;
	virtual auto backward(const arma::Mat<T> &) -> void = 0;
	virtual auto accept(NodeVisitor<T> &visitor) -> void = 0;
	virtual auto name() const -> std::string_view = 0;

	template <typename Other> auto as() -> auto * {
		return dynamic_cast<Other *>(this);
	}
	template <typename Other> auto as() const -> const auto * {
		return dynamic_cast<const Other *>(this);
	}

	auto get_gradient() const { return gradient; }

	auto get_all_inputs() const { return std::span(inputs); }
	auto rows() const {
		if (!value)
			throw MissingValueError("Value not set");
		return value->n_rows;
	}
	auto cols() const {
		if (!value)
			throw MissingValueError("Value not set");
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

	auto get_value() const -> const auto & {
		if (!has_value())
			throw MissingValueError("Value not set");
		return *value;
	}

	auto get_value() -> auto & {
		if (!has_value())
			throw MissingValueError("Value not set");
		return *value;
	}
};

} // namespace ExGraf

#pragma once

#include "exgraf/allowed_types.hpp"

#include <armadillo>
#include <optional>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class Node {
public:
	std::vector<Node<T> *> inputs;
	std::vector<Node<T> *> outputs;
	std::optional<arma::Mat<T>> value;
	arma::Mat<T> gradient;

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
	virtual auto compute() -> arma::Mat<T> = 0;
	virtual auto backward(const arma::Mat<T> &) -> void = 0;

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
};

} // namespace ExGraf

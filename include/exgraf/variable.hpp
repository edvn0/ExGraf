#pragma once

#include "exgraf/node.hpp"

#include <fmt/core.h>

namespace ExGraf {

template <AllowedTypes T> class Variable : public Node<T> {
public:
	explicit Variable(const arma::Mat<T> &initial_value) : Node<T>({}) {
		this->value = initial_value;
		this->gradient =
				arma::zeros<arma::Mat<T>>(initial_value.n_rows, initial_value.n_cols);

		info("Variable::constructor - Initial value shape: ({}, {}), Gradient "
				 "initialized to zeros",
				 initial_value.n_rows, initial_value.n_cols);
	}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto name() const -> std::string_view override {
		return this->is_bias() ? "Bias" : "Weight";
	}

	auto forward() -> arma::Mat<T> override {
		info("Variable::forward - Value shape: ({}, {})", this->rows(),
				 this->cols());
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		info("Variable::backward - Gradient shape: ({}, {})", grad.n_rows,
				 grad.n_cols);
		this->gradient += grad; // Accumulate gradient for updates
		info("Variable::backward - Accumulated gradient shape: ({}, {})",
				 this->gradient.n_rows, this->gradient.n_cols);
	}
};

} // namespace ExGraf

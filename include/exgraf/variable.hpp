#pragma once

#include "exgraf/node.hpp"

namespace ExGraf {

template <AllowedTypes T> class Variable : public Node<T> {
public:
	explicit Variable(const arma::Mat<T> &initial_value) : Node<T>({}) {
		this->value = initial_value;
		this->gradient =
				arma::zeros<arma::Mat<T>>(initial_value.n_rows, initial_value.n_cols);
	}

	auto compute() -> arma::Mat<T> override { return *this->value; }

	auto backward(const arma::Mat<T> &grad) -> void override {
		this->gradient += grad; // Accumulate gradient for updates
	}
};

} // namespace ExGraf

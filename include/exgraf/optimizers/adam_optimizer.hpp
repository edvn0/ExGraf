#pragma once

#include "exgraf/optimizers/optimizer.hpp"

#include <armadillo>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class ADAMOptimizer : public Optimizer<T> {
private:
	using Var = Variable<T>;

public:
	~ADAMOptimizer() override = default;
	T learning_rate;
	T beta1;
	T beta2;
	T epsilon;

	std::unordered_map<Var *, arma::Mat<T>> m;
	std::unordered_map<Var *, arma::Mat<T>> v;

	explicit ADAMOptimizer(T lr, T b1, T b2, T eps)
			: learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

	auto step(std::span<Var *> trainable_nodes) -> void override {
		for (Var *node : trainable_nodes) {
			auto &value = node->get_value();
			if (!m.contains(node)) {
				m[node] = arma::zeros<arma::Mat<T>>(value.n_rows, value.n_cols);
			}
			if (!v.contains(node)) {
				v[node] = arma::zeros<arma::Mat<T>>(value.n_rows, value.n_cols);
			}
		}
	}
};

} // namespace ExGraf

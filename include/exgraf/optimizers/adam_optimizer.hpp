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
	T learning_rate{static_cast<T>(0.01)};
	T beta1{static_cast<T>(0.9)};
	T beta2{static_cast<T>(0.999)};
	T epsilon{static_cast<T>(0.0001)};
	std::size_t t = 0;

	std::unordered_map<Var *, arma::Mat<T>> m;
	std::unordered_map<Var *, arma::Mat<T>> v;

	explicit ADAMOptimizer(T lr, T b1 = 0.9, T b2 = 0.999, T eps = 0.0001)
			: learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

	auto step(std::span<Var *> trainable_nodes) -> void override {
		t++;

		for (Var *node : trainable_nodes) {
			auto &value = node->get_value();
			auto &grad = node->get_gradient();

			if (!m.contains(node)) {
				m[node] = arma::zeros<arma::Mat<T>>(value.n_rows, value.n_cols);
			}
			if (!v.contains(node)) {
				v[node] = arma::zeros<arma::Mat<T>>(value.n_rows, value.n_cols);
			}

			m[node] = beta1 * m[node] + (1 - beta1) * grad;

			v[node] = beta2 * v[node] + (1 - beta2) * (grad % grad);

			auto m_hat = m[node] / (1 - std::pow(beta1, static_cast<T>(t)));

			auto v_hat = v[node] / (1 - std::pow(beta2, static_cast<T>(t)));

			value -= learning_rate * m_hat / (arma::sqrt(v_hat) + epsilon);

			// Clear gradients
			node->zero_gradient();
		}
	}
};

} // namespace ExGraf

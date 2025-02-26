#pragma once

#include "exgraf/optimizers/optimizer.hpp"

#include <armadillo>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class SGDOptimizer : public Optimizer<T> {
private:
	using Var = Variable<T>;
	T learning_rate;

public:
	explicit SGDOptimizer(T lr = T(0.01)) : learning_rate(lr) {}
	~SGDOptimizer() override = default;

	auto step(std::span<Var *> trainable_nodes) -> void override {
		for (Var *node : trainable_nodes) {
			node->get_value() -= learning_rate * node->get_gradient();
		}
	}
};

} // namespace ExGraf

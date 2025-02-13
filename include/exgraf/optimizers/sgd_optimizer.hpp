#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/optimizer.hpp"

#include <armadillo>
#include <unordered_map>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class SgdOptimizer : public Optimizer<T> {
	T learning_rate;
	std::size_t t;
	struct OptState {
		arma::Mat<T> m{};
		OptState() = default;
		OptState(const arma::Mat<T> &param)
				: m(arma::zeros<arma::Mat<T>>(param.n_rows, param.n_cols)) {}
	};
	std::unordered_map<const Tensor<T> *, OptState> state;

public:
	SgdOptimizer(T lr = 0.0001) : learning_rate(lr), t(0) {}

	auto register_tensor(const Tensor<T> &tensor) -> void {
		if (!state.contains(&tensor)) {
			state.emplace(&tensor, OptState(*tensor.data));
		}
	}

	auto
	step(std::vector<std::reference_wrapper<Tensor<T>>> &parameters) -> void {
		t++;
		apply_linear(parameters);
	}

private:
	auto apply_linear(std::vector<std::reference_wrapper<Tensor<T>>> &parameters)
			-> void {
		for (auto &param_ref : parameters) {
			auto &param = param_ref.get();
			if (!param.grad)
				continue;
			register_tensor(param);
			auto &opt_state = state[&param];
			auto &g = *param.grad->data;
			opt_state.m = g;
			param.data->operator-=(learning_rate * opt_state.m);
		}
	}
};

} // namespace ExGraf
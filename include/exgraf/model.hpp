#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/binary_operation.hpp"
#include "exgraf/expression_graph.hpp"
#include "exgraf/optimizer.hpp"
#include "exgraf/tensor.hpp"
#include "exgraf/unary_operation.hpp"

namespace ExGraf {

template <AllowedTypes T> class Model {
	ExpressionGraph<T> graph;
	std::vector<std::reference_wrapper<Tensor<T>>> parameters;
	std::unique_ptr<Optimizer<T>> optimizer;
	Tensor<T> W1;
	Tensor<T> W2;

public:
	Model(std::size_t input_dim, std::size_t hidden_dim, std::size_t output_dim,
				std::unique_ptr<Optimizer<T>> opt)
			: W1(arma::randn<arma::Mat<T>>(input_dim, hidden_dim) *
					 std::sqrt(T(2) / input_dim)),
				W2(arma::randn<arma::Mat<T>>(hidden_dim, output_dim) *
					 std::sqrt(T(2) / hidden_dim)),
				optimizer(std::move(opt)) {
		parameters = {W1, W2};
		for (auto &p : parameters)
			optimizer->register_tensor(p);
	}

	auto forward(const Tensor<T> &input) -> Tensor<T> {
		auto h1 = graph.add_operation(std::make_shared<Binary::MatMulOp<T>>(),
																	{input, W1});
		auto h2 = graph.add_operation(std::make_shared<Unary::ReLUOp<T>>(), {h1});
		auto h3 =
				graph.add_operation(std::make_shared<Binary::MatMulOp<T>>(), {h2, W2});
		return graph.add_operation(std::make_shared<Unary::SoftmaxOp<T>>(), {h3});
	}

	auto compute_loss(const Tensor<T> &output, const Tensor<T> &target) -> T {
		auto loss = graph.add_operation(
				std::make_shared<Binary::CrossEntropyLoss<T>>(), {output, target});
		return loss.data->at(0);
	}

	auto backward(Tensor<T> &loss) -> void { graph.backward(loss); }
	auto step() -> void { optimizer->step(parameters); }

	auto zero_grad() -> void {
		for (auto &p : parameters)
			p.get().zero_grad();
	}

	static auto to_one_hot(const arma::Col<std::size_t> &labels,
												 std::size_t classes) -> Tensor<T> {
		arma::Mat<T> one_hot(labels.n_elem, classes, arma::fill::zeros);
		for (std::size_t i = 0; i < labels.n_elem; ++i)
			one_hot(i, labels(i)) = T(1);
		return Tensor<T>(one_hot);
	}
};

} // namespace ExGraf

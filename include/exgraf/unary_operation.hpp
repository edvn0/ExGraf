#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"
#include "exgraf/operation.hpp"
#include "exgraf/tensor.hpp"

namespace ExGraf::Unary {

template <AllowedTypes T> class ReLUOp : public Operation<T> {
	arma::Mat<T> mask;

public:
	auto
	forward(const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> override {
		auto &input = inputs[0].get();
		trace("[ReLUOp forward] input: {}x{}", input.data->n_rows,
					input.data->n_cols);
		mask = arma::ones<arma::Mat<T>>(input.data->n_rows, input.data->n_cols);
		mask.elem(arma::find(*input.data < T(0))).zeros();
		arma::Mat<T> result = (*input.data) % mask;
		trace("[ReLUOp forward] result: {}x{}", result.n_rows, result.n_cols);
		return Tensor<T>(result);
	}

	auto
	backward(const Tensor<T> &grad_output) -> std::vector<Tensor<T>> override {
		trace("[ReLUOp backward] grad_output: {}x{}", grad_output.data->n_rows,
					grad_output.data->n_cols);
		arma::Mat<T> dx = (*grad_output.data) % mask;
		return {Tensor<T>(dx)};
	}
};

template <AllowedTypes T> class SoftmaxOp : public Operation<T> {
	Tensor<T> last_output;

public:
	auto
	forward(const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> override {
		auto &input = inputs[0].get();
		trace("[SoftmaxOp forward] input: {}x{}", input.data->n_rows,
					input.data->n_cols);
		arma::Mat<T> x = *input.data;
		auto x_max = arma::max(x, 1);
		x.each_col() -= x_max;
		arma::Mat<T> exp_x = arma::exp(x);
		auto sum_exp = arma::sum(exp_x, 1);
		exp_x.each_col() /= sum_exp;
		last_output = Tensor<T>(exp_x);
		trace("[SoftmaxOp forward] output: {}x{}", exp_x.n_rows, exp_x.n_cols);
		return last_output;
	}

	auto
	backward(const Tensor<T> &grad_output) -> std::vector<Tensor<T>> override {
		arma::Mat<T> softmax = *last_output.data;
		arma::Mat<T> grad = *grad_output.data;
		arma::Mat<T> dot = grad % softmax;
		arma::Col<T> dot_sum = arma::sum(dot, 1);
		grad.each_col() -= dot_sum;
		arma::Mat<T> grad_input = softmax % grad;
		return {Tensor<T>(grad_input)};
	}
};

} // namespace ExGraf::Unary

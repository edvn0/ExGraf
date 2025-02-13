#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"
#include "exgraf/operation.hpp"

namespace ExGraf::Binary {

template <AllowedTypes T> class MatMulOp : public Operation<T> {
	Tensor<T> last_input1, last_input2;

public:
	auto
	forward(const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> override {
		auto &A = inputs[0].get();
		auto &B = inputs[1].get();
		trace("[MatMulOp forward] A: {}x{}, B: {}x{}", A.data->n_rows,
					A.data->n_cols, B.data->n_rows, B.data->n_cols);
		last_input1 = A;
		last_input2 = B;
		arma::Mat<T> result = (*A.data) * (*B.data);
		trace("[MatMulOp forward] result: {}x{}", result.n_rows, result.n_cols);
		return Tensor<T>(result);
	}

	auto
	backward(const Tensor<T> &grad_output) -> std::vector<Tensor<T>> override {
		trace("[MatMulOp backward] grad_output: {}x{}", grad_output.data->n_rows,
					grad_output.data->n_cols);
		arma::Mat<T> grad_A = (*grad_output.data) * last_input2.data->t();
		arma::Mat<T> grad_B = last_input1.data->t() * (*grad_output.data);
		return {Tensor<T>(grad_A), Tensor<T>(grad_B)};
	}
};

template <AllowedTypes T> class CrossEntropyLoss : public Operation<T> {
	Tensor<T> last_input;
	Tensor<T> last_target;
	T eps = T(1e-12);

public:
	auto
	forward(const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> override {
		assert(inputs.size() == 2);
		auto &input = inputs[0].get();
		auto &target = inputs[1].get();
		trace("[CrossEntropyLoss forward] input: {}x{}, target: {}x{}",
					input.data->n_rows, input.data->n_cols, target.data->n_rows,
					target.data->n_cols);
		last_input = input;
		last_target = target;
		arma::Mat<T> safe_input = arma::clamp(*input.data, eps, T(1) - eps);
		arma::Mat<T> loss = -(*target.data % arma::log(safe_input));
		T total_loss = arma::accu(loss) / input.data->n_rows;
		trace("[CrossEntropyLoss forward] loss: {}", total_loss);
		return Tensor<T>(
				arma::Mat<T>(arma::SizeMat{1, 1}, arma::fill::value(total_loss)));
	}

	auto
	backward(const Tensor<T> &grad_output) -> std::vector<Tensor<T>> override {
		trace("[CrossEntropyLoss backward] grad_output: {}x{}",
					grad_output.data->n_rows, grad_output.data->n_cols);
		T grad_scale = grad_output.data->at(0) / last_input.data->n_rows;
		arma::Mat<T> grad_input =
				(*last_input.data - *last_target.data) * grad_scale;
		return {Tensor<T>(grad_input)};
	}
};

} // namespace ExGraf::Binary
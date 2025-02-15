#pragma once

#include "exgraf/logger.hpp"
#include "exgraf/node.hpp"

#include <concepts>

namespace ExGraf {

template <AllowedTypes T> class Mult : public Node<T> {
public:
	Mult(Node<T> *lhs, Node<T> *rhs) : Node<T>({lhs, rhs}) {}
	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}
	arma::Mat<T> forward() override {
		// For the z = Op(WX+B) we want the placeholder to be RHS

		auto lhs = this->inputs[1]->forward();
		auto rhs = this->inputs[0]->forward();

		info("Mult::forward - LHS shape: ({}, {}), RHS shape: ({}, {})", lhs.n_rows,
				 lhs.n_cols, rhs.n_rows, rhs.n_cols);

		arma::Mat<T> result = lhs * rhs;
		info("Mult::forward - Result shape: ({}, {})", result.n_rows,
				 result.n_cols);

		this->value = result;
		return *this->value;
	}

	void backward(const arma::Mat<T> &grad) override {
		info("Mult::backward - Gradient shape: ({}, {})", grad.n_rows, grad.n_cols);

		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		arma::Mat<T> lhs_grad = grad * rhs.t();
		arma::Mat<T> rhs_grad = lhs.t() * grad;

		info("Mult::backward - LHS gradient shape: ({}, {}), RHS gradient shape: "
				 "({}, {})",
				 lhs_grad.n_rows, lhs_grad.n_cols, rhs_grad.n_rows, rhs_grad.n_cols);

		this->inputs[0]->backward(lhs_grad);
		this->inputs[1]->backward(rhs_grad);
	}

	auto name() const -> std::string_view override { return "Mult"; }
};

template <AllowedTypes T> class Add : public Node<T> {
public:
	Add(Node<T> *lhs, Node<T> *rhs) : Node<T>({lhs, rhs}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	arma::Mat<T> forward() override {
		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		info("Add::forward - LHS shape: ({}, {}), RHS shape: ({}, {})", lhs.n_rows,
				 lhs.n_cols, rhs.n_rows, rhs.n_cols);

		arma::Mat<T> result;

		// **Row Vector Broadcasting (1, N) + (M, N) → (M, N)**
		if (lhs.n_rows == 1 && lhs.n_cols == rhs.n_cols && rhs.n_rows > 1) {
			result = arma::repmat(lhs, rhs.n_rows, 1) + rhs;
		} else if (rhs.n_rows == 1 && rhs.n_cols == lhs.n_cols && lhs.n_rows > 1) {
			result = lhs + arma::repmat(rhs, lhs.n_rows, 1);
		}
		// **Column Vector Broadcasting (M, 1) + (M, N) → (M, N)**
		else if (lhs.n_cols == 1 && lhs.n_rows == rhs.n_rows && rhs.n_cols > 1) {
			result = arma::repmat(lhs, 1, rhs.n_cols) + rhs;
		} else if (rhs.n_cols == 1 && rhs.n_rows == lhs.n_rows && lhs.n_cols > 1) {
			result = lhs + arma::repmat(rhs, 1, lhs.n_cols);
		}
		// **Standard Addition (Matching Dimensions)**
		else if (lhs.n_rows == rhs.n_rows && lhs.n_cols == rhs.n_cols) {
			result = lhs + rhs;
		} else {
			throw std::runtime_error(fmt::format(
					"Incompatible dimensions for Add operation: ({}, {}) and ({}, {})",
					lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols));
		}

		info("Add::forward - Result shape: ({}, {})", result.n_rows, result.n_cols);
		this->value = result;
		return *this->value;
	}

	void backward(const arma::Mat<T> &grad) override {
		info("Add::backward - Gradient shape: ({}, {})", grad.n_rows, grad.n_cols);

		auto lhs_shape = this->inputs[0]->forward();
		auto rhs_shape = this->inputs[1]->forward();

		// Handle row-vector broadcasting (1, N)
		if (lhs_shape.n_rows == 1 && grad.n_rows > 1) {
			arma::Mat<T> summed_grad = arma::sum(grad, 0);
			this->inputs[0]->backward(summed_grad);
		} else {
			this->inputs[0]->backward(grad);
		}

		// Handle column-vector broadcasting (M, 1)
		if (rhs_shape.n_cols == 1 && grad.n_cols > 1) {
			arma::Mat<T> summed_grad = arma::sum(grad, 1);
			this->inputs[1]->backward(summed_grad);
		} else {
			this->inputs[1]->backward(grad);
		}
	}

	auto name() const -> std::string_view override { return "Add"; }
};

template <AllowedTypes T> class ReLU : public Node<T> {
	T clamp{std::numeric_limits<T>::infinity()}; // Ensures correct type usage
	T lower{static_cast<T>(0.05)};

public:
	explicit ReLU(Node<T> *input, T c = std::numeric_limits<T>::infinity())
			: Node<T>({input}), clamp(c) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto input = this->inputs[0]->forward();
		info("ReLU::forward - Input shape: ({}, {})", input.n_rows, input.n_cols);

		arma::Mat<T> result = arma::clamp(input, lower, clamp);
		info("ReLU::forward - Result shape: ({}, {})", result.n_rows,
				 result.n_cols);

		this->value = result;
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		info("ReLU::backward - Gradient shape: ({}, {})", grad.n_rows, grad.n_cols);

		arma::Mat<T> input_values = this->inputs[0]->forward();
		arma::Mat<T> relu_grad =
				arma::conv_to<arma::Mat<T>>::from(input_values > lower);

		arma::Mat<T> input_grad = grad % relu_grad;
		info("ReLU::backward - Input gradient shape: ({}, {})", input_grad.n_rows,
				 input_grad.n_cols);

		this->inputs[0]->backward(input_grad);
	}

	auto name() const -> std::string_view override { return "ReLU"; }
};

// CrossEntropyLoss
template <AllowedTypes T> class CrossEntropyLoss : public Node<T> {
public:
	CrossEntropyLoss(Node<T> *prediction, Node<T> *target)
			: Node<T>({prediction, target}) {}
	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}
	auto forward() -> arma::Mat<T> override {
		auto p = this->inputs[0]->forward();
		auto t = this->inputs[1]->forward();

		info("CrossEntropyLoss::forward - Prediction shape: ({}, {}), Target "
				 "shape: ({}, {})",
				 p.n_rows, p.n_cols, t.n_rows, t.n_cols);

		arma::Mat<T> result = -arma::sum(t % arma::log(p));
		info("CrossEntropyLoss::forward - Loss value: {}", arma::as_scalar(result));

		this->value = result;
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		info("CrossEntropyLoss::backward - Gradient shape: ({}, {})", grad.n_rows,
				 grad.n_cols);

		auto p = this->inputs[0]->forward();
		auto t = this->inputs[1]->forward();

		arma::Mat<T> pred_grad = -grad % (t / p);
		arma::Mat<T> target_grad = -grad % arma::log(p);

		info("CrossEntropyLoss::backward - Prediction gradient shape: ({}, {}), "
				 "Target gradient shape: ({}, {})",
				 pred_grad.n_rows, pred_grad.n_cols, target_grad.n_rows,
				 target_grad.n_cols);

		this->inputs[0]->backward(pred_grad);
		this->inputs[1]->backward(target_grad);
	}

	auto name() const -> std::string_view override { return "CrossEntropyLoss"; }
};

// Softmax (with numerical stability)
// Softmax (with numerical stability & batch processing)
template <AllowedTypes T> class Softmax : public Node<T> {
public:
	explicit Softmax(Node<T> *input) : Node<T>({input}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto x = this->inputs[0]->forward();
		info("Softmax::forward - Input shape: ({}, {})", x.n_rows, x.n_cols);

		// Compute row-wise max for numerical stability
		arma::Mat<T> x_max = arma::repmat(arma::max(x, 1), 1, x.n_cols);
		arma::Mat<T> exp_x = arma::exp(x - x_max);
		arma::Mat<T> sum_exp = arma::repmat(arma::sum(exp_x, 1), 1, x.n_cols);
		arma::Mat<T> result = exp_x / sum_exp;

		info("Softmax::forward - Result shape: ({}, {})", result.n_rows,
				 result.n_cols);

		this->value = result;
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		info("Softmax::backward - Gradient shape: ({}, {})", grad.n_rows,
				 grad.n_cols);

		auto p = this->forward();

		// Compute per-row Jacobian-vector product
		arma::Mat<T> input_grad(grad.n_rows, grad.n_cols);
		for (arma::uword i = 0; i < grad.n_rows; ++i) {
			arma::Mat<T> d = arma::diagmat(p.row(i)) - p.row(i).t() * p.row(i);
			input_grad.row(i) = grad.row(i) * d;
		}

		info("Softmax::backward - Input gradient shape: ({}, {})",
				 input_grad.n_rows, input_grad.n_cols);

		this->inputs[0]->backward(input_grad);
	}

	auto name() const -> std::string_view override { return "Softmax"; }
};

} // namespace ExGraf

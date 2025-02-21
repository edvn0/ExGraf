#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/logger.hpp"
#include "exgraf/node.hpp"

namespace ExGraf {

struct IncompatibleDimensionsError : std::logic_error {
	template <AllowedTypes T>
	IncompatibleDimensionsError(const arma::Mat<T> &lhs, const arma::Mat<T> &rhs)
			: std::logic_error(fmt::format("Incompatible dimensions for matrix "
																		 "multiplication: ({}, {}) and ({}, {})",
																		 lhs.n_rows, lhs.n_cols, rhs.n_rows,
																		 rhs.n_cols)) {}
};

template <AllowedTypes T> class Hadamard : public Node<T> {
public:
	Hadamard(Node<T> *lhs, Node<T> *rhs)
			: Node<T>(NodeType::Hadamard, {lhs, rhs}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	arma::Mat<T> forward() override {
		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		trace("Hadamard::forward - LHS shape: ({}, {}), RHS shape: ({}, {})",
					lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols);

		if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols) {
			throw IncompatibleDimensionsError(lhs, rhs);
		}

		arma::Mat<T> result = lhs % rhs;

		trace("Hadamard::forward - Result shape: ({}, {})", result.n_rows,
					result.n_cols);
		this->value = std::move(result);
		return *this->value;
	}

	void backward(const arma::Mat<T> &grad) override {
		trace("Hadamard::backward - Gradient shape: ({}, {})", grad.n_rows,
					grad.n_cols);

		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		if (lhs.n_rows != rhs.n_rows || lhs.n_cols != rhs.n_cols) {
			throw IncompatibleDimensionsError(lhs, rhs);
		}

		arma::Mat<T> lhs_grad = grad % rhs;
		arma::Mat<T> rhs_grad = lhs % grad;

		trace("Hadamard::backward - LHS gradient shape: ({}, {}), RHS gradient "
					"shape: "
					"({}, {})",
					lhs_grad.n_rows, lhs_grad.n_cols, rhs_grad.n_rows, rhs_grad.n_cols);

		this->inputs[0]->backward(lhs_grad);
		this->inputs[1]->backward(rhs_grad);
	}

	auto name() const -> std::string_view override { return "Hadamard"; }
};

template <AllowedTypes T> class Mult : public Node<T> {
public:
	Mult(Node<T> *lhs, Node<T> *rhs) : Node<T>(NodeType::Mult, {lhs, rhs}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	arma::Mat<T> forward() override {
		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		trace("Mult::forward - LHS shape: ({}, {}), RHS shape: ({}, {})",
					lhs.n_rows, lhs.n_cols, rhs.n_rows, rhs.n_cols);

		arma::Mat<T> broadcasted_lhs = lhs;
		arma::Mat<T> broadcasted_rhs = rhs;

		if (lhs.n_rows == 1 && rhs.n_rows > 1) {
			broadcasted_lhs = arma::repmat(lhs, rhs.n_rows, 1);
		} else if (rhs.n_cols == 1 && lhs.n_cols > 1) {
			broadcasted_rhs = arma::repmat(rhs, 1, lhs.n_cols);
		}

		if (broadcasted_lhs.n_cols != broadcasted_rhs.n_rows) {
			throw IncompatibleDimensionsError(broadcasted_lhs, broadcasted_rhs);
		}

		arma::Mat<T> result = broadcasted_lhs * broadcasted_rhs;

		trace("Mult::forward - Result shape: ({}, {})", result.n_rows,
					result.n_cols);
		this->value = std::move(result);
		return *this->value;
	}

	void backward(const arma::Mat<T> &grad) override {
		trace("Mult::backward - Gradient shape: ({}, {})", grad.n_rows,
					grad.n_cols);

		auto lhs = this->inputs[1]->forward();
		auto rhs = this->inputs[0]->forward();

		// Handle broadcasting in backward pass
		arma::Mat<T> lhs_grad;
		arma::Mat<T> rhs_grad;

		// Standard matrix multiplication gradients
		if (lhs.n_rows == 1 && rhs.n_rows > 1) {
			// If LHS was broadcasted
			lhs_grad = arma::sum(grad * rhs.t(), 0);
			rhs_grad = lhs.t() * grad;
		} else if (rhs.n_cols == 1 && lhs.n_cols > 1) {
			// If RHS was broadcasted
			lhs_grad = grad * rhs.t();
			rhs_grad = arma::sum(lhs.t() * grad, 1);
		} else {
			// Standard case - no broadcasting
			lhs_grad = grad.t() * rhs;
			rhs_grad = lhs * grad.t();
		}

		trace("Mult::backward - LHS gradient shape: ({}, {}), RHS gradient shape: "
					"({}, {})",
					lhs_grad.n_rows, lhs_grad.n_cols, rhs_grad.n_rows, rhs_grad.n_cols);

		this->inputs[0]->backward(lhs_grad);
		this->inputs[1]->backward(rhs_grad);
	}

	auto name() const -> std::string_view override { return "Mult"; }
};

template <AllowedTypes T> class Add : public Node<T> {
public:
	Add(Node<T> *lhs, Node<T> *rhs) : Node<T>(NodeType::Add, {lhs, rhs}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	arma::Mat<T> forward() override {
		auto lhs = this->inputs[0]->forward();
		auto rhs = this->inputs[1]->forward();

		trace("Add::forward - LHS shape: ({}, {}), RHS shape: ({}, {})", lhs.n_rows,
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
			throw IncompatibleDimensionsError(lhs, rhs);
		}

		trace("Add::forward - Result shape: ({}, {})", result.n_rows,
					result.n_cols);
		this->value = std::move(result);
		return *this->value;
	}

	void backward(const arma::Mat<T> &grad) override {
		trace("Add::backward - Gradient shape: ({}, {})", grad.n_rows, grad.n_cols);

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
			: Node<T>(NodeType::ReLU, {input}), clamp(c) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto input = this->inputs[0]->forward();
		trace("ReLU::forward - Input shape: ({}, {})", input.n_rows, input.n_cols);

		arma::Mat<T> result;
		if (input.n_elem == 0) {
			throw std::runtime_error("Empty input matrix");
		}

		result = input;
		result.transform([this](T val) {
			if (val < lower)
				return lower;
			if (val > clamp)
				return clamp;
			return val;
		});

		trace("ReLU::forward - Result shape: ({}, {})", result.n_rows,
					result.n_cols);
		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		trace("ReLU::backward - Gradient shape: ({}, {})", grad.n_rows,
					grad.n_cols);

		auto input_values = this->inputs[0]->forward();

		if (grad.n_rows != input_values.n_rows ||
				grad.n_cols != input_values.n_cols) {
			throw IncompatibleDimensionsError(input_values, grad);
		}

		arma::Mat<T> relu_grad(input_values.n_rows, input_values.n_cols);
		relu_grad.transform([this](T val) { return val > lower ? 1 : 0; });

		arma::Mat<T> input_grad;
		if (grad.n_elem == 1) {
			input_grad = grad(0, 0) * relu_grad;
		} else {
			input_grad = grad % relu_grad;
		}

		trace("ReLU::backward - Input gradient shape: ({}, {})", input_grad.n_rows,
					input_grad.n_cols);

		this->inputs[0]->backward(input_grad);
	}

	auto name() const -> std::string_view override { return "ReLU"; }
};

template <AllowedTypes T> class SumAxis : public Node<T> {
	int axis;

public:
	SumAxis(Node<T> *input, int a = -1)
			: Node<T>(NodeType::Sum, {input}), axis(a) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto x = this->inputs[0]->forward();
		arma::Mat<T> result;
		if (axis == 0) {
			result = arma::sum(x, 0);
			if (result.n_rows != 1)
				result = result.t();
		} else if (axis == 1) {
			result = arma::sum(x, 1);
			if (result.n_cols != 1)
				result = result.t();
		} else {
			T total = arma::accu(x);
			result.set_size(1, 1);
			result(0, 0) = total;
		}
		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		auto x = this->inputs[0]->forward();
		arma::Mat<T> dx;
		if (axis == 0) {
			dx = arma::repmat(grad, x.n_rows, 1);
		} else if (axis == 1) {
			dx = arma::repmat(grad, 1, x.n_cols);
		} else {
			dx = arma::ones<arma::Mat<T>>(x.n_rows, x.n_cols) * grad(0, 0);
		}
		this->inputs[0]->backward(dx);
	}

	auto name() const -> std::string_view override { return "SumAxis"; }
};

template <AllowedTypes T> class Log : public Node<T> {
public:
	explicit Log(Node<T> *input) : Node<T>(NodeType::Log, {input}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto x = this->inputs[0]->forward();
		trace("Log::forward - Input shape: ({}, {})", x.n_rows, x.n_cols);
		arma::Mat<T> result = arma::log(x);
		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		auto x = this->inputs[0]->forward();
		arma::Mat<T> dx = grad / x;
		this->inputs[0]->backward(dx);
	}

	auto name() const -> std::string_view override { return "Log"; }
};

template <AllowedTypes T> class Neg : public Node<T> {
public:
	explicit Neg(Node<T> *input) : Node<T>(NodeType::Negate, {input}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto x = this->inputs[0]->forward();
		trace("Neg::forward - Input shape: ({}, {})", x.n_rows, x.n_cols);
		arma::Mat<T> result = -x;
		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		arma::Mat<T> dx = -grad;
		this->inputs[0]->backward(dx);
	}

	auto name() const -> std::string_view override { return "Neg"; }
};

template <AllowedTypes T> class CrossEntropyLoss : public Node<T> {
public:
	CrossEntropyLoss(Node<T> *prediction, Node<T> *target)
			: Node<T>(NodeType::CrossEntropyLoss, {prediction, target}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto p = this->inputs[0]->forward();
		auto t = this->inputs[1]->forward();

		trace("CrossEntropyLoss::forward - Prediction shape: ({}, {}), Target "
					"shape: ({}, {})",
					p.n_rows, p.n_cols, t.n_rows, t.n_cols);

		arma::Mat<T> aligned;
		if (p.n_rows == t.n_cols && p.n_cols == t.n_rows) {
			aligned = t.t();
			trace("CrossEntropyLoss::forward - Transposing target matrix to match "
						"prediction shape");
		} else if (p.n_rows == t.n_rows && p.n_cols == t.n_cols) {
			aligned = t;
		} else {
			throw IncompatibleDimensionsError(p, t);
		}

		arma::Mat<T> pointwise_loss = -aligned % arma::log(p);
		T total_loss = arma::accu(pointwise_loss); // Sum all elements
		arma::Mat<T> result(1, 1);
		result(0, 0) = total_loss;

		trace("CrossEntropyLoss::forward - Loss value: {}", total_loss);
		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		trace("CrossEntropyLoss::backward - Gradient shape: ({}, {})", grad.n_rows,
					grad.n_cols);

		auto p = this->inputs[0]->forward();
		auto t = this->inputs[1]->forward();

		bool needs_transpose = (p.n_rows == t.n_cols && p.n_cols == t.n_rows);
		arma::Mat<T> aligned = needs_transpose ? t.t() : t;

		arma::Mat<T> pred_grad = -(aligned / p);
		arma::Mat<T> target_grad = -arma::log(p);

		if (needs_transpose) {
			target_grad = target_grad.t();
		}

		trace("CrossEntropyLoss::backward - Prediction gradient shape: ({}, {}), "
					"Target gradient shape: ({}, {})",
					pred_grad.n_rows, pred_grad.n_cols, target_grad.n_rows,
					target_grad.n_cols);

		this->inputs[0]->backward(pred_grad);
		this->inputs[1]->backward(target_grad);
	}

	auto name() const -> std::string_view override { return "CrossEntropyLoss"; }
};

template <AllowedTypes T> class Softmax : public Node<T> {
public:
	explicit Softmax(Node<T> *input) : Node<T>(NodeType::Softmax, {input}) {}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto forward() -> arma::Mat<T> override {
		auto x = this->inputs[0]->forward();
		trace("Softmax::forward - Input shape: ({}, {})", x.n_rows, x.n_cols);

		// Compute row-wise max for numerical stability
		arma::Mat<T> x_max = arma::repmat(arma::max(x, 1), 1, x.n_cols);
		arma::Mat<T> exp_x = arma::exp(x - x_max);
		arma::Mat<T> sum_exp = arma::repmat(arma::sum(exp_x, 1), 1, x.n_cols);
		arma::Mat<T> result = exp_x / sum_exp;

		trace("Softmax::forward - Result shape: ({}, {})", result.n_rows,
					result.n_cols);

		this->value = std::move(result);
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		trace("Softmax::backward - Gradient shape: ({}, {})", grad.n_rows,
					grad.n_cols);
		arma::Mat<T> S = *this->value; // Softmax output from forward pass

		arma::Mat<T> input_grad(grad.n_rows, grad.n_cols);
		for (arma::uword i = 0; i < grad.n_rows; ++i) {
			arma::Row<T> grad_S = grad.row(i) % S.row(i);
			input_grad.row(i) =
					grad_S - S.row(i) * arma::accu(grad.row(i) % S.row(i));
		}

		trace("Softmax::backward - Input gradient shape: ({}, {})",
					input_grad.n_rows, input_grad.n_cols);
		this->inputs[0]->backward(input_grad);
	}

	auto name() const -> std::string_view override { return "Softmax"; }
};

} // namespace ExGraf

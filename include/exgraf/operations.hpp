#pragma once

#include "exgraf/node.hpp"
#include <concepts>

namespace ExGraf {

template <AllowedTypes T> class Mult : public Node<T> {
public:
	Mult(Node<T> *lhs, Node<T> *rhs) : Node<T>({lhs, rhs}) {}

	arma::Mat<T> compute() override {
		return this->inputs[0]->compute() * this->inputs[1]->compute();
	}

	void backward(const arma::Mat<T> &grad) override {
		this->inputs[0]->backward(grad * this->inputs[1]->compute().t());
		this->inputs[1]->backward(this->inputs[0]->compute().t() * grad);
	}
};

template <AllowedTypes T> class Add : public Node<T> {
public:
	Add(Node<T> *lhs, Node<T> *rhs) : Node<T>({lhs, rhs}) {}

	arma::Mat<T> compute() override {
		return this->inputs[0]->compute() + this->inputs[1]->compute();
	}

	void backward(const arma::Mat<T> &grad) override {
		this->inputs[0]->backward(grad);
		this->inputs[1]->backward(grad);
	}
};

template <AllowedTypes T> class ReLU : public Node<T> {
	double clamp{arma::datum::inf};

public:
	explicit ReLU(Node<T> *input, std::floating_point auto c = arma::datum::inf)
			: Node<T>({input}), clamp(c) {}

	explicit ReLU(Node<T> *input) : Node<T>({input}) {}

	auto compute() -> arma::Mat<T> override {
		return arma::clamp(this->inputs[0]->compute(), 0.0, clamp);
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		arma::Mat<T> relu_grad = this->compute();
		relu_grad.transform([](T val) { return val > 0 ? 1.0 : 0.0; });
		this->inputs[0]->backward(grad % relu_grad);
	}
};

// CrossEntropyLoss
template <AllowedTypes T> class CrossEntropyLoss : public Node<T> {
public:
	CrossEntropyLoss(Node<T> *prediction, Node<T> *target)
			: Node<T>({prediction, target}) {}

	auto compute() -> arma::Mat<T> override {
		auto p = this->inputs[0]->compute();
		auto t = this->inputs[1]->compute();
		return -arma::sum(t % arma::log(p));
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		auto p = this->inputs[0]->compute();
		auto t = this->inputs[1]->compute();
		this->inputs[0]->backward(-grad % (t / p));
		this->inputs[1]->backward(-grad % arma::log(p));
	}
};

// Softmax (with numerical stability)
template <AllowedTypes T> class Softmax : public Node<T> {
public:
	explicit Softmax(Node<T> *input) : Node<T>({input}) {}

	auto compute() -> arma::Mat<T> override {
		arma::Mat x = this->inputs[0]->compute();
		auto exp_x = arma::exp(x - arma::max(x));
		return exp_x / arma::sum(exp_x);
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		auto p = this->compute();
		auto d = arma::diagmat(p) - p * p.t();
		this->inputs[0]->backward(grad * d);
	}
};

} // namespace ExGraf

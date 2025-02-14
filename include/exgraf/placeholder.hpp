#pragma once

#include "exgraf/node.hpp"

#include <string>
#include <string_view>

namespace ExGraf {

struct PlaceholderError : public std::runtime_error {
	using runtime_error::runtime_error;
};

template <AllowedTypes T> class Placeholder : public Node<T> {
public:
	explicit Placeholder(const std::string_view n) : Node<T>({}), name(n) {}

	auto set_value(const arma::Mat<T> &val) { this->value = val; }

	auto compute() -> arma::Mat<T> override {
		if (!this->value)
			throw PlaceholderError("Placeholder value not set");
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		this->gradient = grad;
	}

private:
	std::string name;
};

} // namespace ExGraf

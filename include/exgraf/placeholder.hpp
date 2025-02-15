#pragma once

#include "exgraf/logger.hpp"
#include "exgraf/node.hpp"
#include "exgraf/node_visitor.hpp"

#include <string>
#include <string_view>

namespace ExGraf {

struct PlaceholderError : public std::runtime_error {
	using runtime_error::runtime_error;
};

template <AllowedTypes T> class Placeholder : public Node<T> {
public:
	explicit Placeholder(const std::string_view n)
			: Node<T>({}), placeholder_name(n) {
		info("Placeholder::constructor - Created placeholder with name: {}",
				 name());
	}

	auto set_value(const arma::Mat<T> &val) {
		this->value = val;
		info("Placeholder::set_value - Set value for '{}' with shape: ({}, {})",
				 name(), val.n_rows, val.n_cols);
	}

	auto forward() -> arma::Mat<T> override {
		if (!this->value)
			throw PlaceholderError("Placeholder value not set for " +
														 std::string(name()));

		info("Placeholder::forward - '{}' value shape: ({}, {})", name(),
				 this->rows(), this->cols());
		return *this->value;
	}

	auto backward(const arma::Mat<T> &grad) -> void override {
		info("Placeholder::backward - '{}' gradient shape: ({}, {})", name(),
				 grad.n_rows, grad.n_cols);
		this->gradient = grad;
	}

	auto accept(NodeVisitor<T> &visitor) -> void override {
		visitor.visit(*this);
	}

	auto name() const -> std::string_view override {
		return std::string_view{placeholder_name};
	}

private:
	std::string placeholder_name;
};

} // namespace ExGraf

#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/tensor.hpp"

#include <functional>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class Operation {
public:
	virtual auto
	forward(const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> = 0;
	virtual auto
	backward(const Tensor<T> &grad_output) -> std::vector<Tensor<T>> = 0;
	virtual ~Operation() = default;
};

} // namespace ExGraf

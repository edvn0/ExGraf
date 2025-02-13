#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/tensor.hpp"

#include <functional>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class Optimizer {
public:
	virtual ~Optimizer() = default;
	virtual auto register_tensor(const Tensor<T> &) -> void = 0;
	virtual auto
	step(std::vector<std::reference_wrapper<Tensor<T>>> &) -> void = 0;
};

} // namespace ExGraf

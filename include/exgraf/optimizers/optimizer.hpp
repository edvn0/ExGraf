#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/variable.hpp"

#include <armadillo>
#include <vector>

namespace ExGraf {

template <AllowedTypes T> class Optimizer {
public:
	virtual ~Optimizer() = default;
	virtual auto step(std::span<Variable<T> *>) -> void = 0;
};

} // namespace ExGraf

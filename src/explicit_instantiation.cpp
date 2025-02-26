#include "exgraf/exgraf_pch.hpp"

#include "exgraf/allowed_types.hpp"
#include "exgraf/expression_graph.hpp"
#include "exgraf/operations.hpp"
#include "exgraf/optimizers/adam_optimizer.hpp"
#include "exgraf/optimizers/sgd_optimizer.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/sequential.hpp"
#include "exgraf/variable.hpp"
#include "exgraf/visitors/graphviz.hpp"

namespace ExGraf {
#define X(T) template class Variable<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class Placeholder<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T)                                                                   \
	template class Mult<T>;                                                      \
	template class Add<T>;                                                       \
	template class ReLU<T>;                                                      \
	template class CrossEntropyLoss<T>;                                          \
	template class Softmax<T>;                                                   \
	template class Neg<T>;                                                       \
	template class SumAxis<T>;                                                   \
	template class Hadamard<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class ExpressionGraph<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class GraphvizVisitor<T>;
EXGRAF_ALLOWED_TYPES
#undef X

// Optimizers
#define X(T) template class SGDOptimizer<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class ADAMOptimizer<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class Sequential<T>;
EXGRAF_ALLOWED_TYPES
#undef X

} // namespace ExGraf

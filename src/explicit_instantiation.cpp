#include "exgraf/allowed_types.hpp"
#include "exgraf/expression_graph.hpp"
#include "exgraf/operations.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/variable.hpp"

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
	template class Softmax<T>;
EXGRAF_ALLOWED_TYPES
#undef X

#define X(T) template class ExpressionGraph<T>;
EXGRAF_ALLOWED_TYPES
#undef X

} // namespace ExGraf

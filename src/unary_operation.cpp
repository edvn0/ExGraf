#include "exgraf/unary_operation.hpp"

namespace ExGraf::Unary {

#define X(T)                                                                   \
	template class ReLUOp<T>;                                                    \
	template class SoftmaxOp<T>;

EXGRAF_ALLOWED_TYPES

#undef X

} // namespace ExGraf::Unary

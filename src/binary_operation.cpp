#include "exgraf/binary_operation.hpp"

namespace ExGraf::Binary {

#define X(T)                                                                   \
  template class MatMulOp<T>;                                                  \
  template class CrossEntropyLoss<T>;

EXGRAF_ALLOWED_TYPES

#undef X

} // namespace ExGraf::Binary
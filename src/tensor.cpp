#include "exgraf/tensor.hpp"

namespace ExGraf {

#define X(T) template class Tensor<T>;
EXGRAF_ALLOWED_TYPES
#undef X

} // namespace ExGraf
#include "exgraf/model.hpp"

namespace ExGraf {

#define X(T) template class Model<T>;
EXGRAF_ALLOWED_TYPES
#undef X

} // namespace ExGraf

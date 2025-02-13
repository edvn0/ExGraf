#include "exgraf/expression_graph.hpp"

namespace ExGraf {

#define X(name) template class ExpressionGraph<name>;
EXGRAF_ALLOWED_TYPES
#undef X

} // namespace ExGraf
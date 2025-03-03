#pragma once

#include "exgraf/allowed_types.hpp"

#include "exgraf/node_types.hpp"

namespace ExGraf {

#define X(T) template <AllowedTypes U> class T;
EXGRAF_NODE_LIST_FORWARD;
#undef X
template <AllowedTypes T> class Optimizer;

template <AllowedTypes T> class ExpressionGraph;

} // namespace ExGraf

#pragma once

#include "exgraf/allowed_types.hpp"

namespace ExGraf {

template <AllowedTypes T> class Node;
template <AllowedTypes T> class Placeholder;
template <AllowedTypes T> class Variable;
template <AllowedTypes T> class Add;
template <AllowedTypes T> class Mult;
template <AllowedTypes T> class ReLU;
template <AllowedTypes T> class Tanh;
template <AllowedTypes T> class Softmax;
template <AllowedTypes T> class CrossEntropyLoss;
template <AllowedTypes T> class SumAxis;
template <AllowedTypes T> class Log;
template <AllowedTypes T> class Neg;
template <AllowedTypes T> class Hadamard;

template <AllowedTypes T> class Optimizer;

template <AllowedTypes T> class ExpressionGraph;

} // namespace ExGraf

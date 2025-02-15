#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/forward.hpp"

namespace ExGraf {

// X-Macro List: Define all node types here
#define EXGRAF_NODE_LIST(T)                                                    \
	X(Placeholder<T>)                                                            \
	X(Variable<T>)                                                               \
	X(Add<T>)                                                                    \
	X(Mult<T>)                                                                   \
	X(ReLU<T>)                                                                   \
	X(Softmax<T>)                                                                \
	X(CrossEntropyLoss<T>)

// Visitor Interface Using X-Macro
template <AllowedTypes T> class NodeVisitor {
public:
#define X(NodeType) virtual auto visit(NodeType &) -> void = 0;
	EXGRAF_NODE_LIST(T)
#undef X

	virtual ~NodeVisitor() = default;
};

} // namespace ExGraf

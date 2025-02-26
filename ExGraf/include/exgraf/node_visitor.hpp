#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/forward.hpp"

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace ExGraf {

// X-Macro List: Define all node types here
#define EXGRAF_NODE_LIST(T)                                                    \
	X(Placeholder<T>)                                                            \
	X(Variable<T>)                                                               \
	X(Add<T>)                                                                    \
	X(Mult<T>)                                                                   \
	X(ReLU<T>)                                                                   \
	X(Tanh<T>)                                                                   \
	X(Softmax<T>)                                                                \
	X(CrossEntropyLoss<T>)                                                       \
	X(SumAxis<T>)                                                                \
	X(Log<T>)                                                                    \
	X(Neg<T>)                                                                    \
	X(Hadamard<T>)

// Visitor Interface Using X-Macro
template <AllowedTypes T> class NodeVisitor {
public:
#define X(NodeType) virtual auto visit(NodeType &) -> void = 0;
	EXGRAF_NODE_LIST(T)
#undef X

	virtual ~NodeVisitor() = default;
};

template <AllowedTypes T> class SimpleVisitor : public NodeVisitor<T> {
private:
	std::unordered_set<const Node<T> *> visited;
	std::function<void(Node<T> &)> func;

public:
	explicit SimpleVisitor(std::function<void(Node<T> &)> f)
			: func(std::move(f)) {}

#define X(NodeType)                                                            \
	void visit(NodeType &node) override {                                        \
		if (visited.contains(&node))                                               \
			return;                                                                  \
		for (auto *input : node.get_all_inputs()) {                                \
			input->accept(*this);                                                    \
		}                                                                          \
		visited.insert(&node);                                                     \
		func(node);                                                                \
	}
	EXGRAF_NODE_LIST(T)
#undef X
};

template <AllowedTypes T> auto make_visitor(std::function<void(Node<T> &)> f) {
	return SimpleVisitor<T>(std::move(f));
}

} // namespace ExGraf

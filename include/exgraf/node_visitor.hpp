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

template <AllowedTypes T> class TopologicalVisitor : public NodeVisitor<T> {
public:
	~TopologicalVisitor() = default;
#define X(NodeType)                                                            \
	void visit(NodeType &node) override { visit_recursively(&node); }
	EXGRAF_NODE_LIST(float)
#undef X

	virtual auto do_for_each() -> std::function<void(const Node<T> *)> = 0;

private:
	std::unordered_set<const Node<T> *> visited;

	auto visit_recursively(const Node<T> *node) {
		if (!node || visited.contains(node)) {
			return;
		}
		visited.insert(node);

		for (const auto *input : node->get_all_inputs()) {
			visit_recursively(input);
		}
	}

	auto for_each_in_topological_order(auto &&func) {
		std::ranges::for_each(visited, func);
	}
};

} // namespace ExGraf

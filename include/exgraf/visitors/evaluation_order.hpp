#pragma once

#include "exgraf/logger.hpp"
#include "exgraf/node.hpp"
#include "exgraf/node_visitor.hpp"

#include <unordered_set>

namespace ExGraf {

template <AllowedTypes T> class GraphEvaluationVisitor : public NodeVisitor<T> {
private:
	std::unordered_set<const Node<T> *> visited;

	void visit_recursively(const Node<T> *node) {
		if (!node || visited.contains(node)) {
			return;
		}
		visited.insert(node);

		for (const auto *input : node->get_all_inputs()) {
			visit_recursively(input);
		}

		info("Evaluating Node: {}", node->name());
	}

public:
#define X(NodeType)                                                            \
	void visit(NodeType &node) override { visit_recursively(&node); }
	EXGRAF_NODE_LIST(T)
#undef X
};
} // namespace ExGraf

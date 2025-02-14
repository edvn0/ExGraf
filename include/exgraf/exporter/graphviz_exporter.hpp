#include "exgraf/expression_graph.hpp"

#include <fmt/core.h>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <taskflow/taskflow.hpp>

namespace ExGraf {

template <AllowedTypes T> class GraphvizExporter {
public:
	static std::string to_dot(const ExpressionGraph<T> &graph,
														const std::string &graph_name = "ExpressionGraph",
														const std::string &view_move = "LR") {
		std::ostringstream dot;
		dot << "digraph " << graph_name << " {\n";
		dot << "  rankdir=" << view_move << ";\n";

		std::unordered_map<Node<T> *, std::string> node_ids;
		int node_count = 0;

		for (const auto &node_ptr : graph.nodes) {
			std::string node_id = fmt::format("node{}", node_count++);
			node_ids[node_ptr.get()] = node_id;
			dot << "  " << node_id << " [label=\"" << get_node_label(*node_ptr)
					<< "\", shape=" << get_node_shape(*node_ptr) << "];\n";
		}

		for (const auto &node_ptr : graph.nodes) {
			for (Node<T> *input_node : node_ptr->inputs) {
				dot << "  " << node_ids[input_node] << " -> "
						<< node_ids[node_ptr.get()] << ";\n";
			}
		}

		dot << "}\n";
		return dot.str();
	}

private:
	static std::string get_node_label(const Node<T> &node) {
		if (dynamic_cast<const Placeholder<T> *>(&node))
			return "Placeholder";
		if (auto *var = dynamic_cast<const Variable<T> *>(&node); var != nullptr) {
			const auto is_bias = var->is_bias();
			return is_bias ? "Bias" : "Weight";
		}
		if (dynamic_cast<const Mult<T> *>(&node))
			return "Mult";
		if (dynamic_cast<const Add<T> *>(&node))
			return "Add";
		if (dynamic_cast<const ReLU<T> *>(&node))
			return "ReLU";
		if (dynamic_cast<const Softmax<T> *>(&node))
			return "Softmax";
		if (dynamic_cast<const CrossEntropyLoss<T> *>(&node))
			return "Loss";
		return "Unknown";
	}

	static std::string get_node_shape(const Node<T> &node) {
		if (dynamic_cast<const Placeholder<T> *>(&node))
			return "ellipse";
		if (dynamic_cast<const Variable<T> *>(&node))
			return "box";
		if (dynamic_cast<const Mult<T> *>(&node))
			return "diamond";
		if (dynamic_cast<const Add<T> *>(&node))
			return "diamond";
		if (dynamic_cast<const ReLU<T> *>(&node))
			return "parallelogram";
		if (dynamic_cast<const Softmax<T> *>(&node))
			return "oval";
		if (dynamic_cast<const CrossEntropyLoss<T> *>(&node))
			return "hexagon";
		return "circle";
	}
};

} // namespace ExGraf

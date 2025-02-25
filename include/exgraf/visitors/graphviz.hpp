#pragma once

#include "exgraf/node.hpp"
#include "exgraf/node_visitor.hpp"
#include "exgraf/operations.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <fmt/core.h>

namespace ExGraf {

enum class VisualisationMode : std::uint8_t {
	TopToBottom,
	BottomToTop,
	LeftToRight,
	RightToLeft,
};

template <AllowedTypes T> class GraphvizVisitor : public NodeVisitor<T> {
private:
	std::stringstream dot_stream;
	std::filesystem::path output_file;
	std::unordered_set<const Node<T> *> visited;
	std::unordered_map<const Node<T> *, std::string> node_ids;
	int next_id = 0;

	std::string get_node_id(const Node<T> *node) {
		if (auto it = node_ids.find(node); it != node_ids.end()) {
			return it->second;
		}
		std::string id = fmt::format("node{}", next_id++);
		node_ids[node] = id;
		return id;
	}

	auto get_shape_label(const Node<T> *node) -> std::string {
		try {
			if (node->has_value()) {
				return "(" + std::to_string(node->rows()) + ", " +
							 std::to_string(node->cols()) + ")";
			}
		} catch (...) {
		}
		return "(?, ?)";
	}
	template <class U>
	static constexpr auto is =
			[](const auto *v) { return dynamic_cast<const U *>(v) != nullptr; };
	auto get_node_color(const Node<T> *node) -> std::string {

		if (node->is_bias()) {
			return "lightblue";
		} else if (node->is_weight()) {
			return "lightgreen";
		} else if (is<Placeholder<T>>(node)) {
			return "gold";
		} else if (is<ReLU<T>>(node)) {
			return "lightcoral";
		} else if (is<Softmax<T>>(node)) {
			return "lightpink";
		} else if (is<CrossEntropyLoss<T>>(node)) {
			return "salmon";
		} else {
			return "white";
		}
	}

	auto process_node(const Node<T> &n) -> void {
		const auto *node = &n;
		if (node == nullptr)
			throw std::runtime_error("What are you doing????");

		if (visited.contains(node)) {
			return;
		}

		visited.insert(node);
		auto id = get_node_id(node);
		auto shape_label = get_shape_label(node);
		auto color = get_node_color(node);

		dot_stream << "  " << id << " [label=\"" << node->name() << "\\n"
							 << shape_label << "\", style=filled, fillcolor=" << color
							 << "];\n";

		for (const auto *input : node->get_all_inputs()) {
			process_node(*input);
			dot_stream << "  " << get_node_id(input) << " -> " << id << ";\n";
		}
	}

	constexpr auto to_string(VisualisationMode m) -> std::string_view {
		switch (m) {
		case VisualisationMode::TopToBottom:
			return std::string_view{"TB"};
		case VisualisationMode::BottomToTop:
			return std::string_view{"BT"};
		case VisualisationMode::LeftToRight:
			return std::string_view{"LR"};
		case VisualisationMode::RightToLeft:
			return std::string_view{"RL"};
			break;
		default:
			std::abort();
		}
	}

public:
	explicit GraphvizVisitor(
			const std::string &filename,
			VisualisationMode mode = VisualisationMode::TopToBottom)
			: output_file(filename) {
		dot_stream << "digraph ExpressionGraph {\n";
		dot_stream << "  node [shape=box, fontname=\"Arial\"];\n";
		dot_stream << "  edge [fontname=\"Arial\"];\n";
		dot_stream << "  rankdir=" << to_string(mode) << ";\n";
	}

#define X(NodeType)                                                            \
	void visit(NodeType &node) override { process_node(node); }
	EXGRAF_NODE_LIST(T)
#undef X

	auto get_dot_string() {
		dot_stream << "}\n";
		return dot_stream.str();
	}

	auto finalise() {
		std::ofstream file(output_file);
		if (file) {
			file << get_dot_string();
			return true;
		} else {
			return false;
		}
	}
};

template <AllowedTypes T> class LayerTableVisitor : public NodeVisitor<T> {
private:
	struct layer_info {
		std::string layer_name;
		std::string shape;
		std::size_t param_count;
	};

	std::vector<layer_info> layers;
	std::unordered_set<const Node<T> *> visited;

	auto is_layerish(const Node<T> &n) -> bool {
		if (n.is_weight() || n.is_bias())
			return true;
		if (dynamic_cast<const Add<T> *>(&n))
			return true;
		if (dynamic_cast<const Mult<T> *>(&n))
			return true;
		return false;
	}

	auto store_node(const Node<T> &n) -> void {
		if (visited.count(&n))
			return;
		visited.insert(&n);
		if (!is_layerish(n))
			return;

		std::string shape_str = "(?, ?)";
		std::size_t param_count = 0;

		try {
			if (n.has_value()) {
				shape_str = "(" + std::to_string(n.rows()) + ", " +
										std::to_string(n.cols()) + ")";
			}
		} catch (...) {
		}

		if (n.is_weight())
			param_count = static_cast<std::size_t>(n.rows() * n.cols());
		else if (n.is_bias())
			param_count = static_cast<std::size_t>(n.cols());

		layers.push_back({std::string{n.name()}, shape_str, param_count});

		for (auto *input : n.get_all_inputs()) {
			store_node(*input);
		}
	}

public:
#define X(NodeType)                                                            \
	void visit(NodeType &node) override { store_node(node); }
	EXGRAF_NODE_LIST(T)
#undef X

	auto finalise() -> void {
		fmt::print("{:<25} {:<15} {:<10}\n", "Layer", "Shape", "Params");
		fmt::print("{}\n", std::string(25 + 15 + 10 + 2, '-'));
		for (auto &l : layers) {
			fmt::print("{:<25} {:<15} {:<10}\n", l.layer_name, l.shape,
								 l.param_count);
		}
	}
};

} // namespace ExGraf

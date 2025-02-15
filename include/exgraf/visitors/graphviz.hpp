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
		auto it = node_ids.find(node);
		if (it != node_ids.end()) {
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

	void process_node(const Node<T> &n) {
		const auto *node = &n;
		if (node == nullptr)
			throw std::runtime_error("What are you doing????");

		if (visited.count(node) > 0) {
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

} // namespace ExGraf

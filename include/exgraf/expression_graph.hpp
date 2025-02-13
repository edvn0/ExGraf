#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/operation.hpp"
#include "exgraf/tensor.hpp"

namespace ExGraf {

template <AllowedTypes T> class ExpressionGraph {
	struct Node {
		std::shared_ptr<Operation<T>> op;
		std::vector<const Tensor<T> *> inputs;
		Tensor<T> output;
	};
	std::vector<Node> nodes;
	std::unordered_map<const Tensor<T> *, std::size_t> tensor_to_node_idx;

public:
	auto add_operation(
			std::shared_ptr<Operation<T>> op,
			const std::vector<std::reference_wrapper<const Tensor<T>>> &inputs)
			-> Tensor<T> {
		std::vector<const Tensor<T> *> ptrs;
		for (auto &inp : inputs)
			ptrs.push_back(&inp.get());
		Node node{op, ptrs, op->forward(inputs)};
		nodes.push_back(node);
		tensor_to_node_idx[&nodes.back().output] = nodes.size() - 1;
		return nodes.back().output;
	}

	auto backward(Tensor<T> &start_tensor) -> void {
		for (auto &node : nodes)
			node.output.zero_grad();
		start_tensor.zero_grad();
		start_tensor.grad->data->ones();
		const auto count = static_cast<std::int32_t>(nodes.size()) - 1;
		for (std::int32_t i = count; i >= 0; --i) {
			auto &node = nodes[i];
			auto grad_outputs = node.op->backward(*node.output.grad);
			update_gradients(node, grad_outputs);
		}
	}

private:
	auto update_gradients(Node &node, const std::vector<Tensor<T>> &grad_outputs)
			-> void {
		const auto size = node.inputs.size();
		for (std::size_t j = 0; j < size; ++j) {
			auto *inp = node.inputs[j];
			auto it = tensor_to_node_idx.find(inp);

			if (it != tensor_to_node_idx.end()) {
				auto &inp_node = nodes[it->second];
				if (!inp_node.output.grad) {
					inp_node.output.grad =
							std::make_shared<Tensor<T>>(inp_node.output.shape);
				}
				*inp_node.output.grad->data += *grad_outputs[j].data;
			}
		}
	}
};

} // namespace ExGraf

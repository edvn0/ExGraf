#include <exgraf.hpp>

using namespace ExGraf;

int main() {
	// input (N, 2), output (N, 1)
	ExpressionGraph<float> graph({784, 100, 100, 10});
	graph.compile_model({

	});

	auto exported = GraphvizExporter<float>::to_dot(graph, "MNIST", "TB");

	{
		std::ofstream out("mnist.dot");
		out << exported;
	}

	return 0;
}

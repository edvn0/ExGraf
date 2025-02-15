#include "exgraf/visitors/evaluation_order.hpp"
#include <exgraf.hpp>

#include <armadillo>
#include <fmt/format.h>

template <typename T> struct fmt::formatter<arma::Mat<T>> {
	char presentation = 'f'; // Default to floating-point format

	constexpr auto parse(format_parse_context &ctx)
			-> format_parse_context::iterator {
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && (*it == 'f' || *it == 'e')) {
			presentation = *it++;
		}
		if (it != end && *it != '}') {
			throw format_error("Invalid format for arma::Mat");
		}
		return it;
	}

	template <typename FormatContext>
	auto format(const arma::Mat<T> &mat, FormatContext &ctx) const
			-> decltype(ctx.out()) {
		auto out = ctx.out();
		fmt::format_to(out, "[{} x {}]\n", mat.n_rows, mat.n_cols);
		for (arma::uword i = 0; i < mat.n_rows; ++i) {
			for (arma::uword j = 0; j < mat.n_cols; ++j) {
				if (presentation == 'e') {
					fmt::format_to(out, "{:12.4e} ", mat(i, j));
				} else {
					fmt::format_to(out, "{:8.4f} ", mat(i, j));
				}
			}
			fmt::format_to(out, "\n");
		}
		return out;
	}
};

using namespace ExGraf;

int main() {
	// input (N, 2), output (N, 1)
	ExpressionGraph<double> graph({784, 100, 256, 300, 1000, 30, 10});
	graph.compile_model({});

	auto &&[images, label] =
			MNIST::load("https://raw.githubusercontent.com/fgnt/mnist/master/"
									"t10k-images-idx3-ubyte.gz",
									"https://raw.githubusercontent.com/fgnt/mnist/master/"
									"t10k-labels-idx1-ubyte.gz");

	graph.visit<GraphvizVisitor<double>>("mnist_other.dot",
																			 VisualisationMode::LeftToRight);
	graph.visit<GraphEvaluationVisitor<double>>();

	auto output = graph.predict(images);
	graph.learn(label);

	return 0;
}

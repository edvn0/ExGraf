#include "exgraf/logger.hpp"
#include "exgraf/node_visitor.hpp"
#include "exgraf/placeholder.hpp"
#include "exgraf/variable.hpp"
#include <armadillo>
#include <doctest/doctest.h>
#include <exgraf.hpp>

using namespace ExGraf;
using Matf = arma::Mat<float>;

template <std::floating_point T>
auto matf_equal(const arma::Mat<T> &a, const arma::Mat<T> &b, T tol = 1e-5f)
		-> bool {
	return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
				 arma::approx_equal(a, b, "absdiff", tol);
}

TEST_CASE("sgemm forward") {
	ExpressionGraph<float> graph({3});
	auto ph = graph.add_placeholder("X");
	auto network = graph.add_layer(ph, 2U, 3U);
	// (N, 2) -> (N, 3). (100, 2)*(2, 3) + (3, 1)
	auto input = arma::Mat<float>(1000, 2, arma::fill::ones);
	graph.get_placeholder("X")->set_value(input);
	auto output = network->forward();

	CHECK(output.n_cols == 3);
	CHECK(output.n_rows == 1000);
}

TEST_CASE("sgemm non-basic forwarrd") {
	static constexpr auto input_columns = 784;
	static constexpr auto expected_colums = 10;
	static constexpr auto expected_rows = 412;
	ExpressionGraph<float> graph(
			{input_columns, 1000, 512, 256, expected_colums});
	graph.compile_model({
			.input_size =
					{
							-1,
							input_columns,
					},
	});
	arma::Mat<float> input;
	input.resize(expected_rows, input_columns);
	input.randn();
	auto output = graph.predict(input);

	CHECK(output.n_cols == expected_colums);
	CHECK(output.n_rows == expected_rows);
}

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

TEST_CASE("Mult forward and backward") {
	ExpressionGraph<float> graph{};
	auto lhs = graph.add_placeholder("lhs");
	auto rhs = graph.add_placeholder("rhs");
	auto mult_node = std::make_unique<Mult<float>>(lhs, rhs);

	lhs->set_value(Matf{
			{2.0, 3.0},
			{
					4.0,
					5.0,
			},
	});
	rhs->set_value(Matf{
			{1.0, 2.0},
			{
					3.0,
					4.0,
			},
	});

	Matf expected{
			{11.0, 16.0},
			{
					19.0,
					28.0,
			},
	};
	Matf result = mult_node->forward();
	CHECK(matf_equal(result, expected));

	Matf grad = Matf(2, 2, arma::fill::ones);
	mult_node->backward(grad);
}

TEST_CASE("Add forward and backward") {
	ExpressionGraph<float> graph{};
	auto lhs = graph.add_placeholder("lhs");
	auto rhs = graph.add_placeholder("rhs");
	auto add_node = std::make_unique<Add<float>>(lhs, rhs);

	lhs->set_value(Matf{
			{1.0, 2.0},
			{
					3.0,
					4.0,
			},
	});
	rhs->set_value(Matf{
			{5.0, 6.0},
			{
					7.0,
					8.0,
			},
	});

	Matf expected{
			{6.0, 8.0},
			{
					10.0,
					12.0,
			},
	};
	Matf result = add_node->forward();
	CHECK(matf_equal(result, expected));
}

TEST_CASE("ReLU forward and backward") {
	ExpressionGraph<float> graph{};
	auto input = graph.add_placeholder("input");
	auto relu_node = std::make_unique<ReLU<float>>(input);

	input->set_value(Matf{
			{-1.0, 2.0},
			{
					3.0,
					-4.0,
			},
	});

	Matf expected{
			{0.05F, 2.0},
			{
					3.0,
					0.05F,
			},
	};
	Matf result = relu_node->forward();
	CHECK(matf_equal(result, expected));
}

TEST_CASE("CrossEntropyLoss forward") {
	ExpressionGraph<float> graph{};
	auto prediction = graph.add_placeholder("prediction");
	auto target = graph.add_placeholder("target");
	auto loss_node =
			std::make_unique<CrossEntropyLoss<float>>(prediction, target);

	prediction->set_value(Matf{
			{0.9F, 0.1F},
			{
					0.2F,
					0.8F,
			},
	});
	target->set_value(Matf{
			{1.0, 0.0},
			{
					0.0,
					1.0,
			},
	});

	Matf loss = loss_node->forward();
	CHECK(loss(0, 0) > 0.0f);
}

TEST_CASE("Softmax forward") {
	ExpressionGraph<float> graph{};
	auto input = graph.add_placeholder("input");
	auto softmax_node = std::make_unique<Softmax<float>>(input);

	input->set_value(Matf{
			{1.0, 2.0, 3.0},
			{
					1.0,
					2.0,
					3.0,
			},
	});

	Matf result = softmax_node->forward();
	CHECK(result.n_rows == 2);
	CHECK(result.n_cols == 3);
	CHECK(result(0, 0) > 0.0f);
	CHECK(result(0, 1) > result(0, 0));
	CHECK(result(0, 2) > result(0, 1));
}

TEST_CASE("Softmax backward") {
	SUBCASE("2x3 matrix input") {
		ExpressionGraph<float> graph{};
		auto input = graph.add_placeholder("input");
		auto softmax_node = std::make_unique<Softmax<float>>(input);

		// Input matrix
		input->set_value(Matf{{1.0, 2.0, 3.0}, {4.0, 1.0, 2.0}});

		// Forward pass
		Matf forward_result = softmax_node->forward();

		// Upstream gradient
		Matf grad(2, 3, arma::fill::ones);

		// Backward pass
		softmax_node->backward(grad);

		// Expected result calculated by hand:
		// For row 1: [0.090, 0.245, 0.665]
		// For row 2: [0.709, 0.107, 0.184]
		// Jacobian-vector product for each row
		Matf expected_grad{{0.0816F, 0.1850F, 0.2220F},
											 {0.2054F, 0.0956F, 0.1510F}};

		// Get the actual gradient from the input placeholder
		auto input_grad = input->get_gradient();

		CHECK(matf_equal(input_grad, expected_grad, 1e-4f));
	}

	SUBCASE("1x3 vector input") {
		ExpressionGraph<float> graph{};
		auto input = graph.add_placeholder("input");
		auto softmax_node = std::make_unique<Softmax<float>>(input);

		// Input vector
		input->set_value(Matf{{1.0, 2.0, 3.0}});

		// Forward pass
		Matf forward_result = softmax_node->forward();

		// Upstream gradient
		Matf grad(1, 3, arma::fill::ones);

		// Backward pass
		softmax_node->backward(grad);

		// Expected result for 1D case
		Matf expected_grad{0.0816F, 0.1850F, 0.2220F};
		expected_grad.resize(1, 3);

		auto input_grad = input->get_gradient();
		CHECK(matf_equal(input_grad, expected_grad, 1e-4f));
	}

	SUBCASE("3x1 vector input") {
		ExpressionGraph<float> graph{};
		auto input = graph.add_placeholder("input");
		auto softmax_node = std::make_unique<Softmax<float>>(input);

		// Input column vector
		Matf vec{1.0, 2.0, 3.0};
		vec.resize(3, 1);
		input->set_value(vec);

		// Forward pass
		Matf forward_result = softmax_node->forward();

		// Upstream gradient
		Matf grad(3, 1, arma::fill::ones);

		// Backward pass
		softmax_node->backward(grad);

		// Expected result for column vector
		Matf expected_grad{
				0.0816F,
				0.1850F,
				0.2220F,
		};
		expected_grad.resize(3, 1);

		auto input_grad = input->get_gradient();
		CHECK(matf_equal(input_grad, expected_grad, 1e-4f));
	}

	SUBCASE("Large matrix input") {
		ExpressionGraph<float> graph{};
		auto input = graph.add_placeholder("input");
		auto softmax_node = std::make_unique<Softmax<float>>(input);

		// Create a larger input matrix (10x5)
		Matf input_matrix(10, 5, arma::fill::randu); // Random uniform values
		input->set_value(input_matrix);

		// Forward pass
		Matf forward_result = softmax_node->forward();

		// Upstream gradient
		Matf grad(10, 5, arma::fill::ones);

		// Backward pass
		softmax_node->backward(grad);

		// Check gradient properties
		auto input_grad = input->get_gradient();

		// Verify gradient shape
		CHECK(input_grad.n_rows == 10);
		CHECK(input_grad.n_cols == 5);

		// Verify gradient properties
		// Sum of each row should be close to 0 (property of softmax gradient)
		Matf row_sums = arma::sum(input_grad, 1).t();
		for (size_t i = 0; i < row_sums.n_elem; ++i) {
			CHECK(std::abs(row_sums(i)) < 1e-4f);
		}
	}

	SUBCASE("Numerical gradient check") {
		ExpressionGraph<float> graph{};
		auto input = graph.add_placeholder("input");
		auto softmax_node = std::make_unique<Softmax<float>>(input);

		// Input matrix
		Matf input_matrix{{0.5, 1.0, 1.5}, {2.0, 2.5, 3.0}};
		input->set_value(input_matrix);

		// Compute analytical gradient
		Matf grad(2, 3, arma::fill::ones);
		softmax_node->forward();
		softmax_node->backward(grad);
		auto analytical_grad = input->get_gradient();

		// Compute numerical gradient
		const float epsilon = 1e-4f;
		Matf numerical_grad(2, 3);

		for (size_t i = 0; i < input_matrix.n_rows; ++i) {
			for (size_t j = 0; j < input_matrix.n_cols; ++j) {
				// Positive perturbation
				Matf pos_input = input_matrix;
				pos_input(i, j) += epsilon;
				input->set_value(pos_input);
				Matf pos_output = softmax_node->forward();
				float pos_sum = arma::accu(pos_output);

				// Negative perturbation
				Matf neg_input = input_matrix;
				neg_input(i, j) -= epsilon;
				input->set_value(neg_input);
				Matf neg_output = softmax_node->forward();
				float neg_sum = arma::accu(neg_output);

				// Compute numerical gradient
				numerical_grad(i, j) = (pos_sum - neg_sum) / (2 * epsilon);
			}
		}

		// Compare analytical and numerical gradients
		CHECK(matf_equal(analytical_grad, numerical_grad, 1e-3f));
	}
}

TEST_CASE("sgemm") {
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

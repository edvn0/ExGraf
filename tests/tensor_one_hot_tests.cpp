#include <armadillo>
#include <doctest/doctest.h>
#include <vector>

#include "exgraf/model.hpp"

using namespace ExGraf;

TEST_CASE("to_one_hot produces correct dimensions") {
  using T = double;
  arma::Col<std::size_t> labels = {0, 2, 1};
  std::size_t classes = 3;
  auto result = ExGraf::Model<T>::to_one_hot(labels, classes);

  CHECK_EQ(result.data->n_rows, 3);
  CHECK_EQ(result.data->n_cols, 3);
}

TEST_CASE("to_one_hot sets the correct one-hot entries") {
  using T = double;
  arma::Col<std::size_t> labels = {0, 2, 1};
  std::size_t classes = 3;
  auto result = ExGraf::Model<T>::to_one_hot(labels, classes);

  // Row 0 => label=0 => one_hot(0,0) = 1
  CHECK_EQ((*result.data)(0, 0), doctest::Approx(1.0));
  CHECK_EQ((*result.data)(0, 1), doctest::Approx(0.0));
  CHECK_EQ((*result.data)(0, 2), doctest::Approx(0.0));

  // Row 1 => label=2 => one_hot(1,2) = 1
  CHECK_EQ((*result.data)(1, 0), doctest::Approx(0.0));
  CHECK_EQ((*result.data)(1, 1), doctest::Approx(0.0));
  CHECK_EQ((*result.data)(1, 2), doctest::Approx(1.0));

  // Row 2 => label=1 => one_hot(2,1) = 1
  CHECK_EQ((*result.data)(2, 0), doctest::Approx(0.0));
  CHECK_EQ((*result.data)(2, 1), doctest::Approx(1.0));
  CHECK_EQ((*result.data)(2, 2), doctest::Approx(0.0));
}

TEST_CASE("to_one_hot handles zero-size input") {
  using T = float;
  arma::Col<std::size_t> labels; // empty
  labels.set_size(0);
  std::size_t classes = 10;
  auto result = ExGraf::Model<T>::to_one_hot(labels, classes);

  CHECK_EQ(result.data->n_rows, 0);
  CHECK_EQ(result.data->n_cols, 10);
}

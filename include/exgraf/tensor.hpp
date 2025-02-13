#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/forward.hpp"
#include "exgraf/shape.hpp"

#include <armadillo>
#include <memory>

namespace ExGraf {

template <AllowedTypes T> class Tensor {
public:
  std::shared_ptr<arma::Mat<T>> data{};
  Shape shape{};
  std::shared_ptr<Operation<T>> grad_op{};
  std::shared_ptr<Tensor<T>> grad{};

  Tensor() = default;
  explicit Tensor(const Shape &s)
      : shape(s),
        data(std::make_shared<arma::Mat<T>>(s.dims()[0], s.dims()[1])) {}
  explicit Tensor(const arma::Mat<T> &matrix)
      : shape({matrix.n_rows, matrix.n_cols}),
        data(std::make_shared<arma::Mat<T>>(matrix)) {}

  auto operator[](std::size_t i) -> T & { return (*data)(i); }
  auto operator[](std::size_t i) const -> const T & { return (*data)(i); }

  auto operator()(std::size_t i, std::size_t j) -> T & { return (*data)(i, j); }
  auto operator()(std::size_t i, std::size_t j) const -> const T & {
    return (*data)(i, j);
  }

  auto zero_grad() -> void {
    if (!grad)
      grad = std::make_shared<Tensor<T>>(shape);
    grad->data->zeros();
  }
};
} // namespace ExGraf
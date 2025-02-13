#pragma once

#include "exgraf/allowed_types.hpp"
#include "exgraf/optimizer.hpp"

namespace ExGraf {

template <AllowedTypes T> class AdamOptimizer : public Optimizer<T> {
  T learning_rate, beta1, beta2, epsilon;
  std::size_t t;
  struct OptState {
    arma::Mat<T> m{}, v{};
    OptState() = default;
    OptState(const arma::Mat<T> &param)
        : m(arma::zeros<arma::Mat<T>>(param.n_rows, param.n_cols)),
          v(arma::zeros<arma::Mat<T>>(param.n_rows, param.n_cols)) {}
  };
  std::unordered_map<const Tensor<T> *, OptState> state;

public:
  AdamOptimizer(T lr = 0.0001, T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

  auto register_tensor(const Tensor<T> &tensor) -> void {
    if (!state.contains(&tensor)) {
      state.emplace(&tensor, OptState(*tensor.data));
    }
  }

  auto step(std::vector<std::reference_wrapper<Tensor<T>>> &parameters)
      -> void {
    t++;
    T bias_correction1 = T(1) - std::pow(beta1, T(t));
    T bias_correction2 = T(1) - std::pow(beta2, T(t));
    apply_linear(bias_correction1, bias_correction2, parameters);
  }

private:
  auto apply_linear(const T &bias_correction1, const T &bias_correction2,
                    std::vector<std::reference_wrapper<Tensor<T>>> &parameters)
      -> void {
    for (auto &param_ref : parameters) {
      auto &param = param_ref.get();
      if (!param.grad)
        continue;
      register_tensor(param);
      auto &opt_state = state[&param];
      auto &g = *param.grad->data;
      opt_state.m = beta1 * opt_state.m + (T(1) - beta1) * g;
      opt_state.v = beta2 * opt_state.v + (T(1) - beta2) * (g % g);
      auto m_hat = opt_state.m / bias_correction1;
      auto v_hat = opt_state.v / bias_correction2;
      *param.data -= learning_rate * m_hat / (arma::sqrt(v_hat) + epsilon);
    }
  }
};

} // namespace ExGraf
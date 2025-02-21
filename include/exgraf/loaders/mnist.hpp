#pragma once

#include "exgraf/filesystem.hpp"
#include "exgraf/http/client.hpp"
#include <armadillo>
#include <exception>
#include <string>

#include <taskflow/taskflow.hpp>

namespace ExGraf::MNIST {

auto load(const std::string_view images, const std::string_view labels)
		-> std::pair<arma::Mat<double>, arma::Mat<double>>;

} // namespace ExGraf::MNIST

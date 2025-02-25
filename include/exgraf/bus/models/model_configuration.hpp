#pragma once

#include "exgraf/messaging/serialisable.hpp"

namespace ExGraf::Bus::Models {

struct ModelConfiguration {
	std::string name;
	std::vector<std::size_t> layers;
	std::string optimizer;
	double learning_rate;
};

} // namespace ExGraf::Bus::Models

template <>
struct ExGraf::Messaging::Serializer<ExGraf::Bus::Models::ModelConfiguration> {
	static auto to_json(const Bus::Models::ModelConfiguration &obj)
			-> std::string;
};

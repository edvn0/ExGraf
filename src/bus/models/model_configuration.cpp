#include "exgraf/exgraf_pch.hpp"

#include "exgraf/bus/models/model_configuration.hpp"

#include <boost/json.hpp>
#include <boost/json/array.hpp>

namespace E = ExGraf::Bus::Models;

auto ExGraf::Messaging::Serializer<E::ModelConfiguration>::to_json(
		const E::ModelConfiguration &msg) -> std::string {
	boost::json::object obj;
	obj["name"] = msg.name;
	obj["layers"] = boost::json::array{};
	for (const auto &layer : msg.layers) {
		obj["layers"].emplace_uint64() = layer;
	}
	obj["optimizer"] = msg.optimizer;
	obj["learningRate"] = msg.learning_rate;
	const auto serialised = boost::json::serialize(obj);
	return serialised;
}

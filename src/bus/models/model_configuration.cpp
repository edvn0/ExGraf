#include "exgraf/exgraf_pch.hpp"

#include "exgraf/bus/models/model_configuration.hpp"

#include <boost/json.hpp>
#include <boost/json/array.hpp>

namespace ExGraf::Bus::Models {

auto ModelConfiguration::to_json(const ModelConfiguration &msg) -> std::string {
	boost::json::object obj;
	obj["name"] = msg.name;
	obj["layers"] = boost::json::array{};
	for (const auto &layer : msg.layers) {
		obj["layers"].emplace_uint64() = layer;
	}
	obj["optimizer"] = msg.optimizer;
	obj["learningRate"] = msg.learning_rate;
	return boost::json::serialize(obj);
}

auto ModelConfiguration::from_json(const std::string &json)
		-> ModelConfiguration {
	auto obj = boost::json::parse(json).as_object();
	ModelConfiguration msg;
	msg.name = obj.at("name").as_string();
	std::vector<std::size_t> layers;
	for (const auto &layer : obj.at("layers").as_array()) {
		layers.push_back(static_cast<std::size_t>(layer.as_int64()));
	}
	msg.layers = layers;
	msg.optimizer = obj.at("optimizer").as_string();
	msg.learning_rate = obj.at("learningRate").as_double();
	return msg;
}

} // namespace ExGraf::Bus::Models

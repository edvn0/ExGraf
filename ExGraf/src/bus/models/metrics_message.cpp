#include "exgraf/exgraf_pch.hpp"

#include "exgraf/bus/models/metrics_message.hpp"
#include "exgraf/bus/models/model_configuration.hpp"

#include <boost/json.hpp>

namespace E = ExGraf::Bus::Models;

auto ExGraf::Messaging::Serializer<E::MetricsMessage>::to_json(
		const E::MetricsMessage &msg) -> std::string {
	boost::json::object obj;
	obj["epoch"] = msg.epoch;
	obj["loss"] = msg.loss;
	obj["accuracy"] = msg.accuracy;
	obj["meanPPV"] = msg.mean_ppv;
	obj["meanFPR"] = msg.mean_fpr;
	obj["meanRecall"] = msg.mean_recall;

	if (msg.model_configuration) {
		obj["modelConfiguration"] = boost::json::parse(
				Serializer<E::ModelConfiguration>::to_json(*msg.model_configuration));
	}

	return boost::json::serialize(obj);
}

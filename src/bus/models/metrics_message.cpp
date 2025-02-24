#include "exgraf/exgraf_pch.hpp"

#include "exgraf/bus/models/metrics_message.hpp"
#include "exgraf/bus/models/model_configuration.hpp"

#include <boost/json.hpp>

namespace ExGraf::Bus::Models {

auto MetricsMessage::to_json(const MetricsMessage &msg) -> std::string {
	boost::json::object obj;
	obj["epoch"] = msg.epoch;
	obj["loss"] = msg.loss;
	obj["accuracy"] = msg.accuracy;
	obj["meanPPV"] = msg.mean_ppv;
	obj["meanFPR"] = msg.mean_fpr;
	obj["meanRecall"] = msg.mean_recall;

	if (msg.model_configuration) {
		obj["modelConfiguration"] = boost::json::parse(
				ModelConfiguration::to_json(*msg.model_configuration));
	}

	return boost::json::serialize(obj);
}

auto MetricsMessage::from_json(const std::string &json) -> MetricsMessage {
	auto obj = boost::json::parse(json).as_object();
	MetricsMessage msg;
	msg.epoch = static_cast<std::int32_t>(obj.at("epoch").as_int64());
	msg.loss = obj.at("loss").as_double();
	msg.accuracy = obj.at("accuracy").as_double();
	msg.mean_ppv = obj.at("meanPPV").as_double();
	msg.mean_fpr = obj.at("meanFPR").as_double();
	msg.mean_recall = obj.at("meanRecall").as_double();

	// Don't parse model configuration here, as it's a separate message

	return msg;
}

} // namespace ExGraf::Bus::Models

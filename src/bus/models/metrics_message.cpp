#include "exgraf/exgraf_pch.hpp"

#include "exgraf/bus/models/metrics_message.hpp"

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
	return msg;
}

auto operator<<(std::ostream &os, const MetricsMessage &msg) -> std::ostream & {
	auto json = MetricsMessage::to_json(msg);
	return os << json;
}

} // namespace ExGraf::Bus::Models

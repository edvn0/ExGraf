#pragma once

#include "exgraf/messaging/serialisable.hpp"

namespace ExGraf::Bus::Models {

struct ModelConfiguration;
struct MetricsMessage {
	int epoch{};
	double loss{};
	double accuracy{};
	double mean_ppv{};
	double mean_fpr{};
	double mean_recall{};
	const ModelConfiguration *model_configuration{nullptr};
};

} // namespace ExGraf::Bus::Models

template <>
struct ExGraf::Messaging::Serializer<ExGraf::Bus::Models::MetricsMessage> {
	static auto to_json(const Bus::Models::MetricsMessage &obj) -> std::string;
};

#pragma once

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

	static auto to_json(const MetricsMessage &) -> std::string;
	static auto from_json(const std::string &) -> MetricsMessage;

private:
	friend auto operator<<(std::ostream &os, const MetricsMessage &msg)
			-> std::ostream & {
		auto json = MetricsMessage::to_json(msg);
		return os << json;
	}
};

} // namespace ExGraf::Bus::Models

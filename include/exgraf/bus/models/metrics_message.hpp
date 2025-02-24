#pragma once

namespace ExGraf::Bus::Models {

struct MetricsMessage {
	int epoch{};
	double loss{};
	double accuracy{};
	double mean_ppv{};
	double mean_fpr{};
	double mean_recall{};

	static auto to_json(const MetricsMessage &) -> std::string;
	static auto from_json(const std::string &) -> MetricsMessage;
	friend auto operator<<(std::ostream &os, const MetricsMessage &msg)
			-> std::ostream &;
};

} // namespace ExGraf::Bus::Models

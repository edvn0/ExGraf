#pragma once

namespace ExGraf::Bus::Models {

struct ModelConfiguration {
	std::string name;
	std::vector<std::size_t> layers;
	std::string optimizer;
	double learning_rate;

	static auto to_json(const ModelConfiguration &) -> std::string;
	static auto from_json(const std::string &) -> ModelConfiguration;

private:
	friend auto operator<<(std::ostream &os, const ModelConfiguration &config)
			-> std::ostream & {
		auto json = ModelConfiguration::to_json(config);
		return os << json;
	}
};

} // namespace ExGraf::Bus::Models

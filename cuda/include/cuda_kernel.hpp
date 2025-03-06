#pragma once

#include "cuda_context.hpp"
#include "cuda_helpers.hpp"

#include <filesystem>
#include <nvrtc.h>

namespace cuda {

struct KernelNotFound : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

template <typename T>
concept HasXYZ = requires(T t) {
	t.x;
	t.y;
	t.z;
};
class Kernel {

public:
	MakeNonCopyNonMove(Kernel);

	explicit Kernel(Context &, const std::filesystem::path &);
	~Kernel();

	struct Metrics {
		float time_taken{};
	};
	template <std::ranges::contiguous_range R>
		requires(std::is_same_v<std::ranges::range_value_t<R>, void *>)
	[[nodiscard]] auto launch_kernel(const std::string &name,
																	 HasXYZ auto const &grid_dim,
																	 HasXYZ auto const &block_dim, R &&args)
			-> Metrics {
		auto arguments = std::span(args);
		auto dims = arma::uvec3{
				grid_dim.x,
				grid_dim.y,
				grid_dim.z,
		};
		auto blocks = arma::uvec3{
				block_dim.x,
				block_dim.y,
				block_dim.z,
		};
		auto func = get_function(name.c_str());
		if (!func) {
			throw KernelNotFound("Could not find kernel with requested name.");
		}

		return run_kernel(func, dims, blocks, arguments);
	}

private:
	std::vector<char> ptx_code;
	Context *context;
	CUmodule cuda_module;

	auto runtime_compile(const std::filesystem::path &path) -> void;
	auto run_kernel(CUfunction, const arma::uvec3 &, const arma::uvec3 &,
									const std::span<void *> args) -> Metrics;
	auto get_function(const char *kernel_name) -> CUfunction;
};

} // namespace cuda

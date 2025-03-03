#include "cuda_kernel.hpp"

namespace cuda {

Kernel::~Kernel() { cuModuleUnload(cuda_module); }

Kernel::Kernel(Context &ctx, const std::filesystem::path &path)
		: context(&ctx) {
	if (!std::filesystem::exists(path)) {
		throw std::invalid_argument(
				fmt::format("Kernel file does not exist: {}", path.string()));
	}

	runtime_compile(path);
}

auto Kernel::get_function(const char *kernel_name) -> CUfunction {
	CUfunction func;
	CUDA_CHECK(cuModuleGetFunction(&func, cuda_module, kernel_name));
	return func;
}

auto Kernel::runtime_compile(const std::filesystem::path &path) -> void {
	if (!std::filesystem::exists(path)) {
		throw std::invalid_argument("Kernel file does not exist.");
	}

	{
		std::ifstream code{path, std::ios::ate};
		if (!code) {
			throw std::runtime_error("Failed to open kernel file.");
		}

		std::size_t size = code.tellg();
		code.seekg(0, std::ios::beg);
		ptx_code.resize(size);
		code.read(ptx_code.data(), size);
	}

	nvrtcProgram prog;
	if (auto result = nvrtcCreateProgram(&prog, ptx_code.data(), path.c_str(), 0,
																			 nullptr, nullptr);
			result != NVRTC_SUCCESS) {
		throw std::runtime_error("Failed to create NVRTC program.");
	}

	std::array<const char *, 2> options = {"--gpu-architecture=compute_75",
																				 "--std=c++14"};
	nvrtcResult compile_result;
	if (compile_result =
					nvrtcCompileProgram(prog, options.size(), options.data());
			compile_result != NVRTC_SUCCESS) {
		info("Failed to compile NVRTC program.");
	}

	std::size_t log_size = 0;
	if (auto result = nvrtcGetProgramLogSize(prog, &log_size);
			result != NVRTC_SUCCESS) {
		error("Failed to get NVRTC program log size.");
	}

	std::vector<char> log(log_size);
	if (auto result = nvrtcGetProgramLog(prog, log.data());
			result != NVRTC_SUCCESS) {
		error("Failed to get NVRTC program log.");
	}

	if (compile_result != NVRTC_SUCCESS) {
		info("Compilation error: {}", log.data());
		throw std::runtime_error("Failed to compile NVRTC program.");
	}

	std::size_t ptx_size = 0;
	if (auto result = nvrtcGetPTXSize(prog, &ptx_size); result != NVRTC_SUCCESS) {
		throw std::runtime_error("Failed to get NVRTC PTX size.");
	}

	ptx_code.resize(ptx_size);
	if (auto result = nvrtcGetPTX(prog, ptx_code.data());
			result != NVRTC_SUCCESS) {
		throw std::runtime_error("Failed to get NVRTC PTX.");
	}

	if (auto result = cuModuleLoadDataEx(&cuda_module, ptx_code.data(), 0, 0, 0);
			result != CUDA_SUCCESS) {
		throw std::runtime_error("Failed to load CUDA module.");
	}

	nvrtcDestroyProgram(&prog);
}

auto Kernel::run_kernel(CUfunction func, const arma::uvec3 &grid_dim,
												const arma::uvec3 &block_dim,
												const std::span<void *> args) -> Metrics {
	// Start time event
	CUevent start, stop;
	auto stream = context->create_stream();
	CUDA_DRIVER_CHECK(cuEventCreate(&start, 0));
	CUDA_DRIVER_CHECK(cuEventCreate(&stop, 0));
	CUDA_DRIVER_CHECK(cuEventRecord(start, stream));

	static constexpr auto x = [](const auto &a) { return a.at(0); };
	static constexpr auto y = [](const auto &a) { return a.at(1); };
	static constexpr auto z = [](const auto &a) { return a.at(2); };

	CUDA_DRIVER_CHECK(cuLaunchKernel(func, x(grid_dim), y(grid_dim), z(grid_dim),
																	 x(block_dim), y(block_dim), z(block_dim), 0,
																	 stream, args.data(), nullptr));

	CUDA_DRIVER_CHECK(cuEventRecord(stop, stream));
	CUDA_DRIVER_CHECK(cuStreamSynchronize(stream));
	CUDA_DRIVER_CHECK(cuEventSynchronize(stop));

	Metrics time{};
	CUDA_DRIVER_CHECK(cuEventElapsedTime_v2(&time.time_taken, start, stop));

	context->destroy_stream(stream);

	return time;
}

} // namespace cuda

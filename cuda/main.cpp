#include <algorithm>
#include <armadillo>
#include <bit>
#include <boost/program_options.hpp>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <nvrtc.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#define EXGRAF_LEAK_NAMESPACE
#include "exgraf/logger.hpp"

static constexpr auto to_string = [](CUresult result) {
	return std::to_underlying(result);
};

#define CUDA_CHECK(call)                                                       \
	do {                                                                         \
		if (auto result = call; result != CUDA_SUCCESS) {                          \
			info("CUDA error: {}", to_string(result));                               \
			throw std::runtime_error(                                                \
					fmt::format("CUDA error: {}", to_string(result)));                   \
		}                                                                          \
	} while (0)

#define MakeNonCopyNonMove(Class)                                              \
	Class(const Class &) = delete;                                               \
	Class &operator=(const Class &) = delete;                                    \
	Class(Class &&) = delete;                                                    \
	Class &operator=(Class &&) = delete

class CudaContext;

class CudaMemory {
public:
	MakeNonCopyNonMove(CudaMemory);

	explicit CudaMemory(CudaContext &, std::size_t s) : size(s) {
		if (size == 0) {
			throw std::invalid_argument("Size must be greater than zero.");
		}
		CUDA_CHECK(cuMemAlloc(&device_ptr, size));
	}

	~CudaMemory() {
		if (device_ptr) {
			cuMemFree(device_ptr);
		}
	}

	auto get_size() const -> std::size_t { return size; }

	template <typename T>
	void copy_to_device(const std::span<const T> host_data,
											std::size_t offset = 0) {
		if (offset + host_data.size_bytes() > size) {
			throw std::out_of_range("Memory write out of bounds.");
		}
		CUDA_CHECK(cuMemcpyHtoD(device_ptr + offset, host_data.data(),
														host_data.size_bytes()));
	}
	template <typename T>
	void copy_to_device(const std::span<T> host_data, std::size_t offset = 0) {
		if (offset + host_data.size_bytes() > size) {
			throw std::out_of_range("Memory write out of bounds.");
		}
		CUDA_CHECK(cuMemcpyHtoD(device_ptr + offset, host_data.data(),
														host_data.size_bytes()));
	}

	template <typename T>
	void copy_to_host(std::span<T> host_data, std::size_t offset = 0) const {
		if (offset + host_data.size_bytes() > size) {
			throw std::out_of_range("Memory read out of bounds.");
		}
		CUDA_CHECK(cuMemcpyDtoH(host_data.data(), device_ptr + offset,
														host_data.size_bytes()));
	}

	auto get_device_ptr() const -> const CUdeviceptr & { return device_ptr; }

private:
	CUdeviceptr device_ptr{0};
	std::size_t size{0};
};

class CudaContext {
public:
	MakeNonCopyNonMove(CudaContext);

	explicit CudaContext() {
		CUDA_CHECK(cuInit(0));
		CUDA_CHECK(cuDeviceGet(&device, 0));
		CUDA_CHECK(cuCtxCreate(&context, 0, device));
	}

	~CudaContext() { cuCtxDestroy(context); }

	auto get_context() const -> CUcontext { return context; }

	auto get_device() const -> CUdevice { return device; }

private:
	CUcontext context;
	CUdevice device;
};

class CudaKernel {

public:
	MakeNonCopyNonMove(CudaKernel);

	explicit CudaKernel(CudaContext &ctx, const std::filesystem::path &path)
			: context(&ctx) {
		if (!std::filesystem::exists(path)) {
			throw std::invalid_argument(
					fmt::format("Kernel file does not exist: {}", path.string()));
		}

		runtime_compile(path);
	}

	auto get_function(const char *kernel_name) -> CUfunction {
		CUfunction func;
		cuModuleGetFunction(&func, cuda_module, kernel_name);
		return func;
	}

	~CudaKernel() { cuModuleUnload(cuda_module); }

private:
	std::vector<char> ptx_code;
	CudaContext *context;
	CUmodule cuda_module;

	auto runtime_compile(const std::filesystem::path &path) -> void {
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
		if (auto result = nvrtcCreateProgram(&prog, ptx_code.data(), path.c_str(),
																				 0, nullptr, nullptr);
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
			throw std::runtime_error("Failed to get NVRTC program log size.");
		}

		std::vector<char> log(log_size);
		if (auto result = nvrtcGetProgramLog(prog, log.data());
				result != NVRTC_SUCCESS) {
			throw std::runtime_error("Failed to get NVRTC program log.");
		}

		if (compile_result != NVRTC_SUCCESS) {
			// Log the compilation error.
			info("Compilation error: {}", log.data());
			throw std::runtime_error("Failed to compile NVRTC program.");
		}

		// Extract PTX
		std::size_t ptx_size = 0;
		if (auto result = nvrtcGetPTXSize(prog, &ptx_size);
				result != NVRTC_SUCCESS) {
			throw std::runtime_error("Failed to get NVRTC PTX size.");
		}

		ptx_code.resize(ptx_size);
		if (auto result = nvrtcGetPTX(prog, ptx_code.data());
				result != NVRTC_SUCCESS) {
			throw std::runtime_error("Failed to get NVRTC PTX.");
		}

		// Load into module
		if (auto result =
						cuModuleLoadDataEx(&cuda_module, ptx_code.data(), 0, 0, 0);
				result != CUDA_SUCCESS) {
			throw std::runtime_error("Failed to load CUDA module.");
		}

		nvrtcDestroyProgram(&prog);
	}
};

// boost program options to locate the kernel file:
auto parse_options(int argc, char **argv)
		-> std::optional<std::filesystem::path> {
	using namespace boost::program_options;
	options_description desc{"Options"};
	desc.add_options()("help,h", "Help screen")(
			"kernel,k", value<std::string>()->default_value("kernels/cuda_kernel.cu"),
			"Path to the CUDA kernel file.");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);

	if (vm.count("help")) {
		std::stringstream ss;
		ss << desc;
		info("{}", ss.str());
		return std::nullopt;
	}

	return std::filesystem::path{vm["kernel"].as<std::string>()};
}
auto main(int argc, char **argv) -> int {
	auto kernel_path = parse_options(argc, argv);
	if (!kernel_path) {
		return 0;
	}

	try {
		CudaContext ctx;
		info("Created CUDA context.");

		// Read some info about the GPU device.
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		info("Device name: {}", prop.name);

		CudaKernel kernel(ctx, *kernel_path);
		auto func = kernel.get_function("add");

		// We now have the offsets, we can pass them to the kernel.
		CudaMemory a(ctx, 100 * sizeof(float));
		CudaMemory b(ctx, 100 * sizeof(float));
		CudaMemory c(ctx, 100 * sizeof(float));

		arma::Mat<float> host_a(10, 10, arma::fill::randu);
		arma::Mat<float> host_b(10, 10, arma::fill::randu);
		arma::Mat<float> host_c(10, 10, arma::fill::zeros);

		a.copy_to_device(std::span(host_a.memptr(), host_a.n_elem));
		b.copy_to_device(std::span(host_b.memptr(), host_b.n_elem));

		auto a_ptr = a.get_device_ptr();
		auto b_ptr = b.get_device_ptr();
		auto result_data_ptr = c.get_device_ptr();
		int N = 10;
		int size = N * N;
		std::array<void *, 4> args = {
				&a_ptr,
				&b_ptr,
				&result_data_ptr,
				&size,
		};

		// Start a stream
		CUstream stream;
		CUDA_CHECK(cuStreamCreate(&stream, 0));

		// Start time event
		CUevent start, stop;
		CUDA_CHECK(cuEventCreate(&start, 0)); // Start event
		CUDA_CHECK(cuEventCreate(&stop, 0));	// Stop event
		CUDA_CHECK(cuEventRecord(start, stream));

		dim3 block(N, N);
		dim3 grid(1, 1);

		CUDA_CHECK(cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y,
															block.z, 0, stream, args.data(), nullptr));

		CUDA_CHECK(cuEventRecord(stop, stream));
		CUDA_CHECK(cuEventSynchronize(stop));

		float time;
		CUDA_CHECK(cuEventElapsedTime(&time, start, stop));
		info("Kernel execution time: {} ms", time);

		arma::Mat<float> result(10, 10, arma::fill::zeros);
		c.copy_to_host(std::span(result.memptr(), result.n_elem));
		std::stringstream ss;
		ss << result;
		info("Result: {}", ss.str());

	} catch (const std::exception &e) {
		error("{}", e.what());
		return 1;
	}
	return 0;
}

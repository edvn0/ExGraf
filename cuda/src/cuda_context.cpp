#include "cuda_context.hpp"

namespace cuda {

Context::Context() {
	CUDA_CHECK(cuInit(0));
	CUDA_CHECK(cuDeviceGet(&device, 0));
	CUDA_CHECK(cuCtxCreate(&context, 0, device));
	info("Created CUDA context.");
	cudaGetDeviceProperties(&prop, 0);
}

Context::~Context() {
	cuCtxDestroy(context);
	cudaDeviceReset();
	info("Destroyed CUDA context.");
}

auto Context::get_context() const -> CUcontext { return context; }

auto Context::create_stream() const -> CUstream {
	CUstream stream;
	CUDA_CHECK(cuStreamCreate(&stream, 0));
	return stream;
}

auto Context::destroy_stream(CUstream &stream) -> void {
	cuStreamDestroy(stream);
}

auto Context::get_device() const -> CUdevice { return device; }

auto Context::print_info() const -> void {
	info("Device name: {}", prop.name);
	info("CUDA Capability: {}.{}", prop.major, prop.minor);
	info("Total global memory: {} MB", prop.totalGlobalMem / (1024 * 1024));
	info("Max threads per block: {}", prop.maxThreadsPerBlock);
	info("Max block dimensions: ({}, {}, {})", prop.maxThreadsDim[0],
			 prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	info("Max grid dimensions: ({}, {}, {})", prop.maxGridSize[0],
			 prop.maxGridSize[1], prop.maxGridSize[2]);
}

} // namespace cuda

#pragma once

#include "cuda_helpers.hpp"

namespace cuda {

class Context {
public:
	MakeNonCopyNonMove(Context);
	explicit Context();
	~Context();

	auto get_context() const -> CUcontext;
	auto create_stream() const -> CUstream;
	auto destroy_stream(CUstream &stream) -> void;
	auto get_device() const -> CUdevice;
	auto get_device_properties() const -> const cudaDeviceProp &;
	auto print_info() const -> void;

private:
	CUcontext context;
	CUdevice device;
	cudaDeviceProp prop;
};

} // namespace cuda

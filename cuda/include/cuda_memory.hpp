#pragma once

#include "cuda_helpers.hpp"

namespace cuda {

class Context;

class Memory {
public:
	MakeNonCopyNonMove(Memory);

	explicit Memory(Context &, std::size_t s) : size(s) {
		if (size == 0) {
			throw std::invalid_argument("Size must be greater than zero.");
		}
		CUDA_CHECK(cuMemAlloc(&device_ptr, size));
	}

	template <HasStaticSize T>
	explicit Memory(Context &ctx, const T &type)
			: Memory(ctx, DeviceAllocatable<T>::S) {
		copy_to_device(std::span(&type, 1));
	}

	template <typename T>
	explicit Memory(Context &ctx, const T &type)
			: Memory(ctx, DeviceAllocatable<T>::size(type)) {
		auto data_span = DeviceAllocatable<T>::as_span(type);
		copy_to_device(data_span);
	}

	~Memory() {
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

	auto get_device_ptr() const -> const auto & { return device_ptr; }

private:
	CUdeviceptr device_ptr{0};
	std::size_t size{0};
};

template <typename T> class MatrixMemory {
public:
	MakeNonCopyNonMove(MatrixMemory);

	explicit MatrixMemory(Context &ctx, std::size_t rows, std::size_t cols)
			: storage(rows, cols), memory(ctx, rows * cols * sizeof(T)) {}

	explicit MatrixMemory(Context &ctx, const arma::Mat<T> &mat)
			: storage(mat), memory(ctx, storage) {
		copy_to_device();
	}

	template <typename... Args>
	explicit MatrixMemory(Context &ctx, Args &&...args)
			: storage(std::forward<Args>(args)...), memory(ctx, storage) {
		copy_to_device();
	}

	auto copy_to_device() {
		memory.copy_to_device(std::span(storage.memptr(), storage.n_elem));
	}
	auto copy_to_host() {
		memory.copy_to_host(std::span(storage.memptr(), storage.n_elem));
	}

	auto get_device_ptr() const -> const auto & {
		return memory.get_device_ptr();
	}
	auto get_cpu_data() const -> const auto & { return storage; }

private:
	arma::Mat<T> storage;
	Memory memory;
};

} // namespace cuda

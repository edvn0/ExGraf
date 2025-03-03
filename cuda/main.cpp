#include <armadillo>
#include <boost/program_options.hpp>
#include <filesystem>

#include "cuda_context.hpp"
#include "cuda_kernel.hpp"
#include "cuda_memory.hpp"

#define EXGRAF_LEAK_NAMESPACE
#include "exgraf/logger.hpp"

auto parse_options(int argc, char **argv)
		-> std::optional<std::filesystem::path> {
	using namespace boost::program_options;
	options_description desc{"Options"};
	desc.add_options()("help,h", "Help screen")(
			"kernel,k", value<std::string>()->default_value("kernels/cuda_kernel.cu"),
			"Path to the cuda:: kernel file.");

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

	cuda::Context ctx;
	ctx.print_info();

	cuda::Kernel kernel(ctx, *kernel_path);
	cuda::MatrixMemory<float> host_a(ctx, 10, 10, arma::fill::randu);
	cuda::MatrixMemory<float> host_b(ctx, 10, 10, arma::fill::randu);
	cuda::MatrixMemory<float> host_c(ctx, 10, 10, arma::fill::zeros);

	auto a_ptr = host_a.get_device_ptr();
	auto b_ptr = host_b.get_device_ptr();
	auto result_data_ptr = host_c.get_device_ptr();
	constexpr int N = 10;
	int size = N * N;
	std::array<void *, 4> args = {
			&a_ptr,
			&b_ptr,
			&result_data_ptr,
			&size,
	};

	dim3 block(N, N);
	dim3 grid(1, 1);

	auto &&[time_taken] = kernel.launch_kernel("add", grid, block, args);
	info("Time taken: {}", time_taken);

	static constexpr auto size_matmul = 900U;

	cuda::MatrixMemory<float> output_matmul_memory{
			ctx,
			size_matmul,
			size_matmul,
			arma::fill::randu,
	};
	cuda::MatrixMemory<float> host_matmul_left{
			ctx,
			size_matmul,
			size_matmul,
			arma::fill::randu,
	};
	cuda::MatrixMemory<float> host_matmul_right{
			ctx,
			size_matmul,
			size_matmul,
			arma::fill::zeros,
	};

	dim3 block_matmul(32, 32);
	dim3 grid_size((size_matmul + block_matmul.x - 1) / block_matmul.x,
								 (size_matmul + block_matmul.y - 1) / block_matmul.y);
	std::array sizes{size_matmul, size_matmul, size_matmul, size_matmul};
	bool is_column_major{true};
	std::array matmul_data{
			(void *)&host_matmul_left.get_device_ptr(),
			(void *)&host_matmul_right.get_device_ptr(),
			(void *)&output_matmul_memory.get_device_ptr(),
			(void *)&sizes.at(0),
			(void *)&sizes.at(1),
			(void *)&sizes.at(2),
			(void *)&sizes.at(3),
			(void *)&is_column_major,
	};
	std::array<double, 100> times_storage{};
	for (auto i : std::views::iota(0U, times_storage.size())) {

		auto &&[matmul_time_taken] =
				kernel.launch_kernel("matmul", grid_size, block_matmul, matmul_data);
		times_storage.at(i) = matmul_time_taken;
	}

	arma::rowvec times{times_storage.data(), times_storage.size()};
	info("Total time: {}s, Average time: {}ms, std: {}ms",
			 arma::sum(times) / 1000.0, arma::mean(times), arma::stddev(times, 1));

	host_c.copy_to_host();
	std::stringstream ss;
	ss << host_c.get_cpu_data();

	info("Are CPU side and GPU cuda:: computation (addition) almost (1e-5, "
			 "absdiff) equal? "
			 "{}",
			 arma::approx_equal(host_c.get_cpu_data(),
													host_a.get_cpu_data() + host_b.get_cpu_data(),
													"absdiff", 1e-5));

	output_matmul_memory.copy_to_host();
	info(
			"Are CPU side and GPU cuda:: computation (matmul) almost (1e-5, absdiff) "
			"equal? "
			"{}",
			arma::approx_equal(host_c.get_cpu_data(),
												 host_a.get_cpu_data() * host_b.get_cpu_data(),
												 "absdiff", 1e-5));

	return 0;
}

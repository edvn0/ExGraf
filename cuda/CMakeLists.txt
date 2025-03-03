cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")

project(cudaDemo LANGUAGES CXX CUDA)

find_package(Boost REQUIRED program_options)
find_package(spdlog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(cpr REQUIRED)
find_package(Taskflow REQUIRED)

find_package(CUDAToolkit REQUIRED)

add_executable(cudaDemo main.cpp src/cuda_context.cpp src/cuda_kernel.cpp)
target_link_libraries(cudaDemo PRIVATE ExGraf
	CUDA::cudart_static CUDA::cuda_driver CUDA::nvrtc_static CUDA::nvrtc_builtins_static CUDA::cublas_static CUDA::cufft_static CUDA::curand_static CUDA::cusolver_static CUDA::cusparse_static
	ExGraf Boost::program_options spdlog::spdlog armadillo::armadillo cpr::cpr Taskflow::Taskflow)
target_include_directories(cudaDemo PRIVATE include)

# -Werror=strict-overflow
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(cudaDemo PRIVATE -Wall -Wextra -Wpedantic -Wno-strict-overflow -Wno-null-dereference)
endif()

add_custom_target(copy_kernels ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/kernels
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/kernels ${CMAKE_CURRENT_BINARY_DIR}/kernels
	COMMENT "Ensuring kernels folder exists and copying files..."
)

file(GLOB_RECURSE KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*)

add_custom_command(
	TARGET copy_kernels
	DEPENDS ${KERNEL_FILES}
	COMMENT "Monitoring kernel files for changes..."
)

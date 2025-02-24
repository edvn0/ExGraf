cmake_minimum_required(VERSION 3.24)
project(ExGraf)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Armadillo REQUIRED)
find_package(spdlog REQUIRED)
find_package(cpr REQUIRED)
find_package(Taskflow REQUIRED)
find_package(doctest REQUIRED)
find_package(Boost REQUIRED)
find_package(cppzmq REQUIRED)
find_package(amqpcpp REQUIRED)
find_package(asio REQUIRED)
find_package(libev REQUIRED)

message(STATUS "ExGraf Build Type: ${CMAKE_BUILD_TYPE}")

set(ExGraf_Version_Major 0)
set(ExGraf_Version_Minor 1)
set(ExGraf_Version_Patch 0)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(ExGraf_Version_Status "-dev")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(ExGraf_Version_Status "")
endif()

# Enable strict compiler warnings
set(COMMON_COMPILE_OPTIONS
	-Wall
	-Wextra
	-Wpedantic
	-Werror
	-Wshadow
	-Wconversion
	-Wsign-conversion
	-Wnon-virtual-dtor
	-Wold-style-cast
	-Woverloaded-virtual
	-Wnull-dereference
	-Wunused
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	list(APPEND COMMON_COMPILE_OPTIONS
		-Wdouble-promotion
		-Wformat=2
		-Wstrict-overflow=5
		-Wundef
		-Wcast-align
		-Wcast-qual
		-Wmisleading-indentation
		-Wduplicated-cond
		-Wduplicated-branches
		-Wlogical-op
		-Wuseless-cast
	)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	list(APPEND COMMON_COMPILE_OPTIONS
		/W4
		/WX
		/permissive-
		/sdl
	)
endif()

add_compile_options(${COMMON_COMPILE_OPTIONS})

set(PROJECT_VERSION
	"${ExGraf_Version_Major}.${ExGraf_Version_Minor}.${ExGraf_Version_Patch}${ExGraf_Version_Status}"
)
message(STATUS "${PROJECT_NAME} version: ${PROJECT_VERSION}")

add_library(ExGrafHttp STATIC src/http/client.cpp)
target_include_directories(ExGrafHttp PUBLIC include)
target_link_libraries(ExGrafHttp PRIVATE cpr::cpr Taskflow::Taskflow)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(ExGrafHttp PUBLIC -Wno-null-dereference -Wno-strict-overflow)
endif()

add_library(
	ExGraf
	STATIC
	src/explicit_instantiation.cpp
	src/filesystem.cpp
	src/loaders/mnist.cpp
	src/messaging/zero_mq_transport.cpp
	src/messaging/rabbit_mq_transport.cpp
	src/bus/models/metrics_message.cpp
)
target_include_directories(ExGraf PUBLIC include)
target_link_libraries(ExGraf PRIVATE ExGrafHttp Armadillo::Armadillo Boost::boost Boost::json asio::asio spdlog::spdlog Taskflow::Taskflow cppzmq amqpcpp libev::libev)
target_precompile_headers(ExGraf PUBLIC include/exgraf/exgraf_pch.hpp)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})

enable_testing()
add_subdirectory(tests)

add_subdirectory(demo)

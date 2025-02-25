cmake_minimum_required(VERSION 3.24)
project(ExGrafDemo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ExGrafDemo main.cpp)
target_link_libraries(ExGrafDemo PRIVATE ExGraf Boost::program_options spdlog::spdlog armadillo::armadillo cpr::cpr Taskflow::Taskflow)

# -Werror=strict-overflow
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(ExGrafDemo PRIVATE -Wall -Wextra -Wpedantic -Wno-strict-overflow -Wno-null-dereference)
endif()

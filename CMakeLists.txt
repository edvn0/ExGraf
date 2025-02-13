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

message(STATUS "ExGraf Build Type: ${CMAKE_BUILD_TYPE}")

set(ALLOW_TASKFLOW ON CACHE BOOL "Allow Taskflow to be used in the project")

set(ExGraf_Version_Major 0)
set(ExGraf_Version_Minor 1)
set(ExGraf_Version_Patch 0)
if (CMAKE_BUILD_TYPE STREQUAL "Dev")
	set(ExGraf_Version_Status "-dev")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(ExGraf_Version_Status "-rel")
endif()
set(PROJECT_VERSION
  "${ExGraf_Version_Major}.${ExGraf_Version_Minor}.${ExGraf_Version_Patch}${ExGraf_Version_Status}"
)
message(STATUS "${PROJECT_NAME} version: ${PROJECT_VERSION}")

add_library(ExGrafHttp STATIC src/http/client.cpp)
target_include_directories(ExGrafHttp PUBLIC include)
target_link_libraries(ExGrafHttp PRIVATE cpr::cpr)

add_executable(
  ExGraf
  src/main.cpp
  src/binary_operation.cpp
  src/unary_operation.cpp
  src/tensor.cpp
  src/model.cpp
  src/expression_graph.cpp
)
target_include_directories(ExGraf PUBLIC include)

target_link_libraries(ExGraf PRIVATE ExGrafHttp Armadillo::Armadillo spdlog::spdlog Taskflow::Taskflow)

if(${Taskflow_FOUND})
  target_compile_definitions(ExGraf PRIVATE HAS_TASKFLOW)
endif()

enable_testing()
add_subdirectory(tests)

cmake_minimum_required(VERSION 3.24)
project(ExGraf)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "ExGraf: Build the tests" ON)

set(ExGraf_Version_Major 0)
set(ExGraf_Version_Minor 1)
set(ExGraf_Version_Patch 0)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(ExGraf_Version_Status "-dev")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(ExGraf_Version_Status "")
endif()

set(PROJECT_VERSION
	"${ExGraf_Version_Major}.${ExGraf_Version_Minor}.${ExGraf_Version_Patch}${ExGraf_Version_Status}"
)
message(STATUS "ExGraf: ${PROJECT_NAME} version: ${PROJECT_VERSION}")

add_subdirectory(ExGraf)
add_subdirectory(demo)

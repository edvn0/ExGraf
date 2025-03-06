add_executable(ExGrafTests
  test_main.cpp
	fs_tests.cpp
	operations_tests.cpp
	basic_graph_tests.cpp
)
target_include_directories(ExGrafTests PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(ExGrafTests
  PRIVATE
  ExGrafHttp
	ExGraf
  Armadillo::Armadillo
  doctest::doctest
  Taskflow::Taskflow
  spdlog::spdlog
)

set(TEST_RESOURCES_DIR ${CMAKE_BINARY_DIR}/test_resources)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fs_tests.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/fs_tests.cpp @ONLY)
file(MAKE_DIRECTORY ${TEST_RESOURCES_DIR})

# Check if you have a data directory, if not create one with test files
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/test)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test DESTINATION ${TEST_RESOURCES_DIR})
else()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/test)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/data/test/sample.txt "Test data for file system tests")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/test DESTINATION ${TEST_RESOURCES_DIR})
endif()

add_test(NAME ExGrafTests COMMAND ExGrafTests)

add_executable(ExGrafTests
  test_main.cpp
)
target_include_directories(ExGrafTests PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(ExGrafTests
  PRIVATE
  ExGrafHttp
  Armadillo::Armadillo
  doctest::doctest
  Taskflow::Taskflow
  spdlog::spdlog
)

add_test(NAME ExGrafTests COMMAND ExGrafTests)

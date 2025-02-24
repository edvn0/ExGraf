from conan import ConanFile
from conan.tools.files import copy
from conan.tools.cmake import CMake, cmake_layout
import os


class ExGrafConan(ConanFile):
    name = "exgraf"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("spdlog/1.15.0")
        self.requires("armadillo/12.6.4", options={"use_hdf5": False})
        self.requires("cpr/1.11.1")
        self.requires("doctest/2.4.11")
        self.requires("taskflow/3.9.0")
        self.requires("boost/1.86.0")
        self.requires("cppzmq/4.10.0")
        self.requires("amqp-cpp/4.3.26")
        self.requires("asio/1.32.0")

    def layout(self) -> None:
        cmake_layout(self)

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

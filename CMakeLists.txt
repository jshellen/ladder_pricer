cmake_minimum_required(VERSION 3.15)
project(ladder_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ladder_pricer cpp/ladder_pricer/bindings.cpp)
target_include_directories(ladder_pricer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ladder_pricer)

install(TARGETS ladder_pricer
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
    RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
)
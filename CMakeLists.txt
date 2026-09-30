cmake_minimum_required(VERSION 3.20)
project(trinity_ladder_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ladder_pricer_core STATIC
    cpp/src/matrix.cpp
    cpp/src/core.cpp
    cpp/src/howard.cpp
    cpp/src/analytics.cpp
    cpp/src/simulation.cpp
)
target_include_directories(ladder_pricer_core PUBLIC cpp/include)
target_compile_options(ladder_pricer_core PRIVATE -Wall -Wextra -Wpedantic)

option(TRINITY_BUILD_TESTS "Build native C++ regression tests" ON)
if(TRINITY_BUILD_TESTS)
    add_executable(test_howard cpp/tests/test_howard.cpp)
    target_link_libraries(test_howard PRIVATE ladder_pricer_core)

    add_executable(test_analytics cpp/tests/test_analytics.cpp)
    target_link_libraries(test_analytics PRIVATE ladder_pricer_core)

    add_executable(test_simulation cpp/tests/test_simulation.cpp)
    target_link_libraries(test_simulation PRIVATE ladder_pricer_core)
endif()

cmake_minimum_required(VERSION 3.20)
project(trinity_ladder_pricer LANGUAGES CXX)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

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
if(APPLE)
    target_link_libraries(ladder_pricer_core PUBLIC ${ACCELERATE_FRAMEWORK})
else()
    target_link_libraries(ladder_pricer_core PUBLIC LAPACK::LAPACK BLAS::BLAS)
endif()
target_compile_options(ladder_pricer_core PRIVATE -Wall -Wextra -Wpedantic)

option(TRINITY_BUILD_TESTS "Build native C++ regression tests" ON)
if(TRINITY_BUILD_TESTS)
    add_executable(test_howard cpp/tests/test_howard.cpp)
    target_link_libraries(test_howard PRIVATE ladder_pricer_core)
    enable_testing()
    add_test(NAME howard_residual_diagnostics COMMAND test_howard)

    add_executable(test_analytics cpp/tests/test_analytics.cpp)
    target_link_libraries(test_analytics PRIVATE ladder_pricer_core)

    add_executable(test_simulation cpp/tests/test_simulation.cpp)
    target_link_libraries(test_simulation PRIVATE ladder_pricer_core)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_brent cpp/tests/test_brent.cpp)
    target_link_libraries(test_brent PRIVATE ladder_pricer_core)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_rfq_pricing_knot_invariance
        cpp/tests/test_rfq_pricing_knot_invariance.cpp)
    target_link_libraries(test_rfq_pricing_knot_invariance PRIVATE ladder_pricer_core)
    enable_testing()
    add_test(NAME rfq_pricing_knot_invariance COMMAND test_rfq_pricing_knot_invariance)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_single_pricing_knot_optimum cpp/tests/test_single_pricing_knot_optimum.cpp)
    target_link_libraries(test_single_pricing_knot_optimum PRIVATE ladder_pricer_core)
    add_test(NAME single_pricing_knot_optimum COMMAND test_single_pricing_knot_optimum)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_single_pricing_knot_global cpp/tests/test_single_pricing_knot_global.cpp)
    target_link_libraries(test_single_pricing_knot_global PRIVATE ladder_pricer_core)
    add_test(NAME single_pricing_knot_global COMMAND test_single_pricing_knot_global)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_flowsource_components cpp/tests/test_flowsource_components.cpp)
    target_link_libraries(test_flowsource_components PRIVATE ladder_pricer_core)
    add_test(NAME flowsource_components COMMAND test_flowsource_components)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_ecn_fxmapping cpp/tests/test_ecn_fxmapping.cpp)
    target_link_libraries(test_ecn_fxmapping PRIVATE ladder_pricer_core)
    add_test(NAME ecn_fxmapping COMMAND test_ecn_fxmapping)
endif()

if(TRINITY_BUILD_TESTS)
    add_executable(test_inventory_instantaneous_moments cpp/tests/test_inventory_instantaneous_moments.cpp)
    target_link_libraries(test_inventory_instantaneous_moments PRIVATE ladder_pricer_core)
    add_test(NAME inventory_instantaneous_moments COMMAND test_inventory_instantaneous_moments)
endif()

cmake_minimum_required(VERSION 3.13)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(phase_v2 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall")
if(CMAKE_CXX_COMPILER MATCHES ".*icpx")
  message("Selected icpx flags")
  set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG")
else()
  message("Selected conventional flags")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release") 
  add_definitions(-DSIM_AVX2)
  add_definitions(-DMATH_AVX2)
endif()

add_library(lib INTERFACE)
target_include_directories(lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(sample_mcmc ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_mcmc.cpp)
target_link_libraries(sample_mcmc PRIVATE lib)

function(create_mytest TEST_NAME)
  add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/test/${TEST_NAME}.cpp)
  target_link_libraries(${TEST_NAME} PRIVATE lib)
  add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)
endfunction()
create_mytest(test_order)
create_mytest(test_collection)
create_mytest(test_sim)

add_executable(perf_eval ${CMAKE_CURRENT_SOURCE_DIR}/test/perf_eval.cpp)
target_link_libraries(perf_eval PRIVATE lib)
target_link_options(perf_eval PRIVATE -pg)
target_compile_options(perf_eval PRIVATE -pg -qopt-report)

add_subdirectory(topics)

set(PYTHON_EXECUTABLE /${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python)
add_subdirectory(pybind11)
pybind11_add_module(opy src/opy.cpp)
target_link_libraries(opy PRIVATE lib)
pybind11_add_module(window_rate src/window_rate.cpp)
target_link_libraries(window_rate PRIVATE lib)

add_custom_target(copy_shared_object ALL
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/build/*.so ${CMAKE_CURRENT_SOURCE_DIR}/py/
  DEPENDS opy window_rate
)

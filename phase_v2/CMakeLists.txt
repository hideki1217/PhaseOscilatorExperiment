cmake_minimum_required(VERSION 3.13)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(phase_v2 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall")
if(CMAKE_CXX_COMPILER MATCHES ".*icpx")
  message("Selected icpx flags")
  set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG -qopenmp")
else()
  message("Selected conventional flags")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fopenmp")
endif()

add_library(lib INTERFACE)
target_include_directories(lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_order ${CMAKE_CURRENT_SOURCE_DIR}/test/test_order.cpp)
target_link_libraries(test_order PRIVATE lib)
add_test(NAME test_order COMMAND $<TARGET_FILE:test_order>)

add_executable(test_collection ${CMAKE_CURRENT_SOURCE_DIR}/test/test_order.cpp)
target_link_libraries(test_collection PRIVATE lib)
add_test(NAME test_collection COMMAND $<TARGET_FILE:test_collection>)

set(PYTHON_EXECUTABLE /${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python)
add_subdirectory(pybind11)
pybind11_add_module(opy src/opy.cpp)
target_link_libraries(opy PRIVATE lib)

add_custom_command(TARGET opy POST_BUILD
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/build/*.so ${CMAKE_CURRENT_SOURCE_DIR}/py/
)
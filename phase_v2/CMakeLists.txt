cmake_minimum_required(VERSION 3.13)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(phase_v2 CXX)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O1 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG -qopenmp")

add_library(lib INTERFACE)
target_include_directories(lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_order ${CMAKE_CURRENT_SOURCE_DIR}/test/test_order.cpp)
target_link_libraries(test_order PRIVATE lib)
add_test(NAME test_order COMMAND $<TARGET_FILE:test_order>)

add_executable(test_collection ${CMAKE_CURRENT_SOURCE_DIR}/test/test_order.cpp)
target_link_libraries(test_collection PRIVATE lib)
add_test(NAME test_collection COMMAND $<TARGET_FILE:test_collection>)

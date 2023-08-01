cmake_minimum_required(VERSION 3.13)
enable_testing()

project(phase CXX)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fast")

add_executable(phase phase.cpp)
target_include_directories(phase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(phase PRIVATE -openmp)

add_custom_target(run 
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
    DEPENDS phase
)

add_custom_target(async_run 
    COMMAND nohup ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
    DEPENDS phase
)



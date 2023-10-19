cmake_minimum_required(VERSION 3.13)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(phase CXX)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS "-fast -DNDEBUG -qopenmp")

add_executable(phase phase.cpp)
target_include_directories(phase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# (defined NEWTON) ? Newton : RK4 
# target_compile_definitions(phase PRIVATE NEWTON)

add_custom_target(run 
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
    DEPENDS phase
)

add_custom_target(async_run 
    COMMAND nohup ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
    DEPENDS phase
)

add_executable(phase_asym phase_asym.cpp)
target_include_directories(phase_asym PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# (defined NEWTON) ? Newton : RK4 
# target_compile_definitions(phase_asym PRIVATE NEWTON)

add_custom_target(async_run_asym 
    COMMAND nohup ${CMAKE_CURRENT_SOURCE_DIR}/run_asym.sh
    DEPENDS phase_asym
)

add_executable(test_math tests/test_math.cpp)
target_include_directories(test_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME test_math
    COMMAND $<TARGET_FILE:test_math>
)

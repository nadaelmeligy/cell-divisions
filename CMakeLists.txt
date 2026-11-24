cmake_minimum_required (VERSION 3.1)
project (tvm)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

aux_source_directory(. DIR_SRCS)
add_subdirectory(Run)
add_subdirectory(Vertex)
add_subdirectory(Edge)
add_subdirectory(Polygon)
add_subdirectory(Cell)
add_subdirectory(Energy)
add_subdirectory(Reconnection)
add_subdirectory(ECM)
add_executable(tvm tvm.cpp)
target_link_libraries(tvm run vertex edge polygon cell energy reconnection ECM)

# CHANGED 2026-11-24: 'ctest' in build/ runs experiments/jen_followup/check_tao_port.py (Tao's 64-cell
# reference, exact restarts, mesh checks). Turn off with -DBUILD_TESTING=OFF.
include(CTest)
if(BUILD_TESTING)
    find_program(TVM_TEST_PYTHON NAMES python3)
    if(TVM_TEST_PYTHON)
        add_executable(tao_reference_check experiments/jen_followup/tao_reference_check.cpp tvm.cpp)
        target_compile_definitions(tao_reference_check PRIVATE TVM_NO_MAIN)
        target_link_libraries(tao_reference_check run vertex edge polygon cell energy reconnection ECM)
        add_test(NAME tao_port
                 COMMAND "${TVM_TEST_PYTHON}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/experiments/jen_followup/check_tao_port.py"
                         $<TARGET_FILE:tvm> $<TARGET_FILE:tao_reference_check>)
    endif()
endif()

cmake_minimum_required(VERSION 2.8)
# 声明一个 cmake 工程
project(${smoke_testcase}_tiling)

add_executable(${smoke_testcase}_tiling 
    ${CMAKE_SOURCE_DIR}/custom_tiling/main.cpp
)

message(STATUS "Ascend install path is : ${ASCEND_INSTALL_PATH}")

target_include_directories(${smoke_testcase}_tiling PRIVATE
    ${ASCEND_INSTALL_PATH}/include/
    ${ASCEND_INSTALL_PATH}/runtime/include/
    ${ASCEND_INSTALL_PATH}/runtime/include/tiling/
)

target_link_directories(${smoke_testcase}_tiling PRIVATE
    ${ASCEND_INSTALL_PATH}/lib64/
    ${ASCEND_INSTALL_PATH}/runtime/lib64/
)

target_compile_options(${smoke_testcase}_tiling PRIVATE
    -g
    -std=c++17
)

target_link_libraries(${smoke_testcase}_tiling PRIVATE
    tiling_api
    -Wl,--no-as-needed
    c_sec
    graph
    register
    -Wl,--as-needed
)

set_target_properties(${smoke_testcase}_tiling PROPERTIES
    OUTPUT_NAME ${smoke_testcase}_tiling
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensures the standard is enforced
set(CMAKE_CXX_EXTENSIONS OFF)       # Disable compiler-specific extensions

set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_INSTALL_PATH} CACHE STRING "ASCEND CANN package installation directory")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/cmake)
set(RUN_MODE "ONBOARD" CACHE STRING "Choose the run mode: ONBOARD ro SIMULATOR.")
set(ASCEND_RUN_MODE ${RUN_MODE})

set(SOC_VERSION "ascend910" CACHE STRING "Choose SoC Version")
set(ASCEND_PRODUCT_TYPE ${SOC_VERSION})

set(ASCEND_CORE_TYPE "AiCore")
set(ASCEND_CUSTOM_TILING "NO_CUSTOM_TILING")

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE)

set(smoke_testcase runner)

set(CMAKE_CCE_SOURCE_FILE_EXTENSIONS cce cpp)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB KERNEL_FILES
    vadd_flags.cce
)
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)

file(GLOB HOST_FILES
    main.cpp
)
set_source_files_properties(${HOST_FILES} PROPERTIES LANGUAGE CCE)

execute_process(
    COMMAND which ${CMAKE_CXX_COMPILER}
    RESULT_VARIABLE WHICH_RESULT
    OUTPUT_VARIABLE GCC_PATH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT WHICH_RESULT EQUAL 0)
    message(FATAL_ERROR "which ${CMAKE_CXX_COMPILER} not found. "
    "Please make sure it is installed and available in the PATH."
    "And set CMAKE_CXX_COMPILER=g++")
endif()

get_filename_component(GCC_PATH "${GCC_PATH}" DIRECTORY)
get_filename_component(GCC_TOOLCHAIN_PATH "${GCC_PATH}" DIRECTORY)

# ===================================================================
add_library(ascendc_kernels SHARED
    ${KERNEL_FILES} vadd.cpp
)
target_compile_definitions(ascendc_kernels PUBLIC
    TILING_KEY_VAR=0
)

target_compile_options(ascendc_kernels PUBLIC
    -O2
    -std=c++17
    -fPIC
    --gcc-toolchain=${GCC_TOOLCHAIN_PATH}
)
target_compile_features(ascendc_kernels PUBLIC cxx_std_17)

# ===================================================================
add_library(${smoke_testcase}_lib_npu SHARED
    ${HOST_FILES}
)
target_compile_features(${smoke_testcase}_lib_npu PUBLIC cxx_std_17)


target_compile_options(${smoke_testcase}_lib_npu PUBLIC
    -O2
    -std=c++17
    -fPIC
)

target_link_directories(${smoke_testcase}_lib_npu PUBLIC
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${smoke_testcase}_lib_npu PUBLIC
    ascendc_kernels
    acl_cblas
)

# ===================================================================
add_executable(${smoke_testcase}_npu
    ${HOST_FILES}
)

target_compile_options(${smoke_testcase}_npu PUBLIC
    -O2
    -std=c++17
    -fPIC
)
target_compile_features(${smoke_testcase}_npu PUBLIC cxx_std_17)

target_compile_definitions(${smoke_testcase}_npu PRIVATE
    TILING_KEY_VAR=0
)

target_link_libraries(${smoke_testcase}_npu PRIVATE
    ${smoke_testcase}_lib_npu
)

# ===================================================================
file(GLOB STUB_FILES
    ${CMAKE_SOURCE_DIR}/stub.cpp
)

add_library(stub_library SHARED
    ${STUB_FILES}
)

target_compile_features(stub_library PUBLIC cxx_std_17)

target_compile_options(stub_library PRIVATE
    -O2
    -std=c++17
    -fPIC
)

# ===================================================================
find_library(ASCENDCL_LIB ascendcl HINTS ${DACE_ASCEND_INSTALL_PATH}/runtime/lib64)
find_library(ASCENDCL_RUNTIME_LIB runtime HINTS ${DACE_ASCEND_INSTALL_PATH}/runtime/lib64)

target_link_libraries(stub_library PUBLIC
    ${smoke_testcase}_lib_npu
    ascendc_kernels
    dl
    ascendcl
    runtime
)
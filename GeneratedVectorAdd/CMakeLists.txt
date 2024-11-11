cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_INSTALL_PATH} CACHE STRING "ASCEND CANN package installation directory")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/cmake)
set(RUN_MODE "ONBOARD" CACHE STRING "Choose the run mode: ONBOARD ro SIMULATOR.")
set(ASCEND_RUN_MODE ${RUN_MODE})

set(SOC_VERSION "ascend910" CACHE STRING "Choose SoC Version")
set(ASCEND_PRODUCT_TYPE ${SOC_VERSION})

set(ASCEND_CORE_TYPE "AiCore")

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(kernel_samples LANGUAGES CCE CXX)

set(smoke_testcase runner)

add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)
/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */

#include "common.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdlib>

struct ascendc_test_3_state_t {
    dace::ascendc::Context *acl_context;
};


extern "C" int __dace_init_ascendc(ascendc_test_3_state_t *__state);
extern "C" int __dace_exit_ascendc(ascendc_test_3_state_t *__state);
extern "C" float __dace_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C);

extern "C" void var_one(int TOTAL_RUNS){

    constexpr int PROGRESS_INTERVAL = 100;
    constexpr size_t ARRAY_SIZE = 8192;
    constexpr size_t DATA_SIZE = ARRAY_SIZE * sizeof(aclFloat16);


    // Create binary file for writing kernel execution times
    std::ofstream binaryFile("empty_kernel_times.bin", std::ios::out | std::ios::binary);
    if (!binaryFile) {
        std::cerr << "Failed to open binary file for writing!" << std::endl;
        return;
    }

    // Kernel state initialization
    ascendc_test_3_state_t __state;
    __dace_init_ascendc(&__state);

    // Allocate host memory (aligned to 64 bytes) and device memory
    aclFloat16 *A = static_cast<aclFloat16*>(std::aligned_alloc(64, DATA_SIZE));
    aclFloat16 *B = static_cast<aclFloat16*>(std::aligned_alloc(64, DATA_SIZE));
    aclFloat16 *C = static_cast<aclFloat16*>(std::aligned_alloc(64, DATA_SIZE));

    aclFloat16 *ascend_A, *ascend_B, *ascend_C;
    DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_A, DATA_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_B, DATA_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_C, DATA_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));

    // Initialize device memory
    DACE_ACL_CHECK(aclrtMemcpy(ascend_A, DATA_SIZE, A, DATA_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));
    DACE_ACL_CHECK(aclrtMemcpy(ascend_B, DATA_SIZE, B, DATA_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));
    DACE_ACL_CHECK(aclrtSynchronizeDevice());

    std::cout << "Starting kernel execution for " << TOTAL_RUNS << " runs..." << std::endl;

    for (int i = 0; i < TOTAL_RUNS; ++i) {
        // Measure the execution time for the kernel
        auto start = std::chrono::high_resolution_clock::now();
        auto kerneltime = __dace_runkernel_copy_map_outer_0_0_6(&__state, reinterpret_cast<uint8_t*>(ascend_A), reinterpret_cast<uint8_t*>(ascend_B), reinterpret_cast<uint8_t*>(ascend_C));
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate and store kernel time
        std::chrono::duration<float> duration = end - start;
        float total_time_ms = duration.count() * 1000.0f;

        // Write the kernel execution time to the binary file
        binaryFile.write(reinterpret_cast<const char*>(&kerneltime), sizeof(kerneltime));

        // Print progress every 100 iterations
        if ((i + 1) % PROGRESS_INTERVAL == 0) {
            std::cout << "Progress: " << (i + 1) << "/" << TOTAL_RUNS << " completed." << std::endl;
        }

        DACE_ACL_CHECK(aclrtSynchronizeDevice());
    }

    std::cout << "Kernel execution completed." << std::endl;

    // Clean up resources
    binaryFile.close();
    DACE_ACL_CHECK(aclrtFree(ascend_A));
    DACE_ACL_CHECK(aclrtFree(ascend_B));
    DACE_ACL_CHECK(aclrtFree(ascend_C));
    std::free(A);
    std::free(B);
    std::free(C);
}


extern "C" int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <total_runs>" << std::endl;
        return 1;
    }

    int TOTAL_RUNS = std::atoi(argv[1]);
    if (TOTAL_RUNS <= 0) {
        std::cerr << "Invalid number of runs specified." << std::endl;
        return 1;
    }

    var_one(TOTAL_RUNS );
    //var_two(TOTAL_RUNS );
}

extern "C"  void __program_ascendc_test_3(ascendc_test_3_state_t *__state)
{
}

extern "C"  int __dace_exit_ascendc_test_3(ascendc_test_3_state_t *__state);

extern "C"  ascendc_test_3_state_t *__dace_init_ascendc_test_3()
{
    int __result = 0;
    ascendc_test_3_state_t *__state = new ascendc_test_3_state_t;


    __result |= __dace_init_ascendc(__state);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

extern "C"  int __dace_exit_ascendc_test_3(ascendc_test_3_state_t *__state)
{
    int __err = 0;

    int __err_ascendc = __dace_exit_ascendc(__state);
    if (__err_ascendc) {
        __err = __err_ascendc;
    }
    delete __state;
    return __err;
}

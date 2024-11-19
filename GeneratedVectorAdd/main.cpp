/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include "common.h"
#include <cstring>
#include <mutex>
struct ascendc_test_3_state_t {
    dace::ascendc::Context *acl_context;
};


int __dace_init_ascendc(ascendc_test_3_state_t *__state);
void __dace_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C);
int main()
{

    ascendc_test_3_state_t __state;
    std::cout << "A00" << std::endl;
    __dace_init_ascendc(&__state);
    std::mutex mtx;
    mtx.lock();
    {
        std::cout << "A0" << std::endl;
        aclFloat16 *A; // 8
        A =  static_cast<aclFloat16*>(std::aligned_alloc(64, 8192 * 8192 * sizeof(aclFloat16)));
        aclFloat16 * ascend_A;
        DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_A, 8192 *  8192 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
        aclFloat16 *B; // 8
        B =  static_cast<aclFloat16*>(std::aligned_alloc(64,8192 *  8192 * sizeof(aclFloat16)));
        aclFloat16 *C; // 8
        C =  static_cast<aclFloat16*>(std::aligned_alloc(64,8192 * 8192 * sizeof(aclFloat16)));

        aclFloat16 * ascend_B;
        aclFloat16 * ascend_C;
        for (size_t i = 0; i < 8192 * 8192; ++i) {
          A[i] = aclFloatToFloat16(1.0f);
          B[i] = aclFloatToFloat16(2.0f);
        }
        bool is_full_of_ones = true;
        for (size_t i = 0; i < 8192 * 8192; ++i) {
            if (aclFloat16ToFloat(A[i]) != 1.0f) {
                is_full_of_ones = false;
                break;
            }
        }
        if (is_full_of_ones) {;
          std::cout << "A is initialized to 1s." << std::endl;
        } else {
          std::cout << "A is NOT initialized to 1s." << std::endl;
        }
        bool is_full_of_twoes = true;
        for (size_t i = 0; i < 8192 * 8192; ++i) {
            if (aclFloat16ToFloat(B[i]) != 2.0f) {
                is_full_of_twoes = false;
                break;
            }
        }
        if (is_full_of_twoes) {;
          std::cout << "B is initialized to 2s." << std::endl;
        } else {
          std::cout << "B is NOT initialized to 2s." << std::endl;
        }
        DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_B, 8192 * 8192 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
        DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_C, 8192 * 8192 * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
        std::cout << "A0.1" << std::endl;
        DACE_ACL_CHECK(aclrtMemcpy(ascend_A, 8192 * 8192 * sizeof(aclFloat16) * 2, A,8192 *  8192 * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
        DACE_ACL_CHECK(aclrtMemcpy(ascend_B, 8192 * 8192 * sizeof(aclFloat16) * 2, B, 8192 * 8192 * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
        std::cout << "A0.2" << std::endl;
        DACE_ACL_CHECK(aclrtSynchronizeDevice());
        std::cout << "A1" << std::endl;
        __dace_runkernel_copy_map_outer_0_0_6(&__state, reinterpret_cast<uint8_t*>(ascend_A), reinterpret_cast<uint8_t*>(ascend_B), reinterpret_cast<uint8_t*>(ascend_C));
        DACE_ACL_CHECK(aclrtMemcpy(C, 8192 *  8192 * sizeof(aclFloat16) * 2, ascend_C, 8192 * 8192 * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));
        DACE_ACL_CHECK(aclrtSynchronizeDevice());
        std::cout << "A2" << std::endl;
        // Check if B is full of 1s after the kernel run
        is_full_of_ones = true;
        int first_index = -1;
        for (size_t i = 0; i < 8192 * 8192; ++i) {
            if (aclFloat16ToFloat(C[i]) != 3.0f) {
                is_full_of_ones = false;
                if (aclFloat16ToFloat(C[i]) == 0 && first_index == -1){
                    first_index = i;
                    std::cout << aclFloat16ToFloat(C[i-1]) << ", ";
                    std::cout << aclFloat16ToFloat(C[i]) << ", ";
                }
                //std::cout << aclFloat16ToFloat(C[i]) << ", ";
            }
        }

        if (is_full_of_ones) {
            std::cout << "C is full of 3s." << std::endl;
        } else {
            std::cout << "C is NOT full of 3s." << std::endl;
            std::cout << "First index: " << first_index << std::endl;
        }
        std::free(A);
        std::free(B);
    }
    mtx.unlock();
}

 void __program_ascendc_test_3(ascendc_test_3_state_t *__state)
{
}
 int __dace_init_ascendc(ascendc_test_3_state_t *__state);
 int __dace_exit_ascendc(ascendc_test_3_state_t *__state);

 ascendc_test_3_state_t *__dace_init_ascendc_test_3()
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

 int __dace_exit_ascendc_test_3(ascendc_test_3_state_t *__state)
{
    int __err = 0;

    int __err_ascendc = __dace_exit_ascendc(__state);
    if (__err_ascendc) {
        __err = __err_ascendc;
    }
    delete __state;
    return __err;
}

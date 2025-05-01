/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#ifdef DACE_ASCEND
#ifndef __CCE_KT_TEST__
#endif
#include "../../include/hash.h"

struct ascendc_test_1_state_t {
    dace::ascendc::Context *acl_context;
};

DACE_EXPORTED void __program_ascendc_test_1_internal(ascendc_test_1_state_t*__state, float * __restrict__ A)
{

    {
        float * ascend_A;
        DACE_ACL_CHECK(aclrtMalloc((void**)&ascend_A, 512 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));

        DACE_ACL_CHECK(aclrtMemcpy(reinterpret_cast<void*>(ascend_A), 512 * sizeof(float), reinterpret_cast<const void*>(A), 512 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        DACE_ACL_CHECK(aclrtSynchronizeDevice());

    }
}

DACE_EXPORTED void __program_ascendc_test_1(ascendc_test_1_state_t *__state, float * __restrict__ A)
{
    __program_ascendc_test_1_internal(__state, A);
}
DACE_EXPORTED int __dace_init_ascendc(ascendc_test_1_state_t *__state);
DACE_EXPORTED int __dace_exit_ascendc(ascendc_test_1_state_t *__state);

DACE_EXPORTED ascendc_test_1_state_t *__dace_init_ascendc_test_1()
{
    int __result = 0;
    ascendc_test_1_state_t *__state = new ascendc_test_1_state_t;


    __result |= __dace_init_ascendc(__state);

    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_ascendc_test_1(ascendc_test_1_state_t *__state)
{
    int __err = 0;

    int __err_ascendc = __dace_exit_ascendc(__state);
    if (__err_ascendc) {
        __err = __err_ascendc;
    }
    delete __state;
    return __err;
}
#ifdef DACE_ASCEND
#endif
#endif

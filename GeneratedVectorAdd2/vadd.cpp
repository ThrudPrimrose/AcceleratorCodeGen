
#include "common.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>

struct ascendc_test_3_state_t
{
    dace::ascendc::Context *acl_context;
};

extern "C" int __dace_init_ascendc(ascendc_test_3_state_t *__state);
extern "C" int __dace_exit_ascendc(ascendc_test_3_state_t *__state);

extern "C"  int __dace_init_ascendc(ascendc_test_3_state_t *__state)
{
    __state->acl_context = new dace::ascendc::Context(1, 1);
    std::cout << "Start initializing" << std::endl;
    DACE_ACL_CHECK(aclInit({}));
    DACE_ACL_CHECK(aclrtSetDevice(7));
    aclrtContext *c = &__state->acl_context->aclrt_context;
    DACE_ACL_CHECK(aclrtCreateContext(c, 7));
    // Initialize acl before we run the application
    float *dev_X;
    DACE_ACL_CHECK(aclrtMalloc((void **)&dev_X, 1, ACL_MEM_MALLOC_HUGE_FIRST));
    DACE_ACL_CHECK(aclrtFree(dev_X));

    // Create acl streams and events
    for (int i = 0; i < 1; ++i)
    {
        DACE_ACL_CHECK(aclrtCreateStream(&__state->acl_context->internal_streams[i]));
        __state->acl_context->streams[i] = __state->acl_context->internal_streams[i]; // Allow for externals to modify streams
    }
    // for(int i = 0; i < 1; ++i) {
    //     DACE_ACL_CHECK(aclrtEventCreateWithFlags(&__state->acl_context->events[i], aclrtEventDisableTiming));
    // }
    std::cout << "Initialization complete" << std::endl;
    return 0;
}

extern "C"  int __dace_exit_ascendc(ascendc_test_3_state_t *__state)
{

    // Destroy aclrt streams and events
    for (int i = 0; i < 1; ++i)
    {
        DACE_ACL_CHECK(aclrtDestroyStream(__state->acl_context->internal_streams[i]));
    }
    // for(int i = 0; i < 1; ++i) {
    //     DACE_ACL_CHECK(aclrtDestroyEvent(__state->acl_context->events[i]));
    // }

    delete __state->acl_context;
    return 0;
}

extern "C"  bool __dace_acl_set_stream(ascendc_test_3_state_t *__state, int streamid, aclrtStream stream)
{
    if (streamid < 0 || streamid >= 1)
    {
        return false;
    }

    __state->acl_context->streams[streamid] = stream;

    return true;
}

extern "C"  void __dace_acl_set_all_streams(ascendc_test_3_state_t *__state, aclrtStream stream)
{
    for (int i = 0; i < 1; ++i)
    {
        __state->acl_context->streams[i] = stream;
    }
}

extern "C" float __dace_run_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C);
extern "C" float __dace_runkernel_copy_map_outer_0_0_6(ascendc_test_3_state_t *__state, uint8_t* ascend_A, uint8_t* ascend_B, uint8_t* ascend_C)
{

    auto start = std::chrono::high_resolution_clock::now();

    __dace_run_runkernel_copy_map_outer_0_0_6(__state, ascend_A, ascend_B, ascend_C);
    DACE_ACL_CHECK(aclrtSynchronizeDevice());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    return duration.count() * 1000.0f ;

}
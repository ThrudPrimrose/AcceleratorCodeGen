/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include <dace/soft_hier/preload.h>
#include "../../include/hash.h"

typedef struct GEMM_state_t {
    int filler;
}GEMM_state_t;

// DACE_EXPORTED void __dace_runkernel_gemm_entry_0_0_0(GEMM_state_t *__state, uint32_t A, uint32_t B, uint32_t C);
void __program_GEMM_internal(GEMM_state_t*__state, unsigned short * __restrict__ A, unsigned short * __restrict__ B, unsigned short * __restrict__ C)
{

    {
        //Framecode generating state main...
        // A = A;
        // B = B;
        printf("Start Running Kernel");
        int result = system("cd ./.dacecache && ./dace.sh");
        printf("Result: %d", result);
        printf("Finish Running Kernel");
        // __dace_runkernel_gemm_entry_0_0_0(__state, A, B, C);
        // C = C;

    }
}

DACE_EXPORTED void __program_GEMM(GEMM_state_t *__state, unsigned short * __restrict__ A, unsigned short * __restrict__ B, unsigned short * __restrict__ C)
{
    __program_GEMM_internal(__state, A, B, C);
}

DACE_EXPORTED GEMM_state_t *__dace_init_GEMM()
{
    int __result = 0;
    GEMM_state_t *__state = new GEMM_state_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_GEMM(GEMM_state_t *__state)
{
    int __err = 0;
    delete __state;
    return __err;
}

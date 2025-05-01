/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct vadd2_state_t {

};

#include <omp.h>
void __program_vadd2_internal(vadd2_state_t*__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int N)
{

    {

        {
            #pragma omp parallel
            {
                auto i = omp_get_thread_num();
                {
                    for (int64_t _ = 0; _ < N; _ += 32) {
                        float __tmp2;
                        {
                            float __in1 = A[i];
                            float __in2 = B[i];
                            float __out;

                            ///////////////////
                            // Tasklet code (_Add_)
                            __out = (__in1 + __in2);
                            ///////////////////

                            __tmp2 = __out;
                        }
                        {
                            float __inp = __tmp2;
                            float __out;

                            ///////////////////
                            // Tasklet code (assign_18_12)
                            __out = __inp;
                            ///////////////////

                            C[i] = __out;
                        }
                    }
                }
            }
        }

    }
}

DACE_EXPORTED void __program_vadd2(vadd2_state_t *__state, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, int N)
{
    __program_vadd2_internal(__state, A, B, C, N);
}

DACE_EXPORTED vadd2_state_t *__dace_init_vadd2(int N)
{
    int __result = 0;
    vadd2_state_t *__state = new vadd2_state_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED int __dace_exit_vadd2(vadd2_state_t *__state)
{
    int __err = 0;
    delete __state;
    return __err;
}

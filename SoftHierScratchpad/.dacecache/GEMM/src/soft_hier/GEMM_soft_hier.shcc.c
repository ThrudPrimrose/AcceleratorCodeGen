
// #include <cuda_runtime.h>
// #include <dace/dace.h>
#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "flex_dump.h"
#define floor(x) ((x))
#define Mod(x, y) ((x) % (y))

typedef struct GEMM_state_t {
    int filler;
}GEMM_state_t;



static uint64_t HBM_ADDRESS_SPACE = 0x20000000;
static uint64_t HBM_ADDRESS_BASE = 0xc0000000;

int __dace_init_cuda(struct GEMM_state_t *__state);
int __dace_exit_cuda(struct GEMM_state_t *__state);

typedef struct DacePlacementInfo
{
    uint32_t channel_id;
    uint32_t tile_offset;
} DacePlacementInfo;

uint32_t A_base;
uint32_t A_height;
uint32_t A_width;
uint32_t A_tile_height;
uint32_t A_tile_width;
const DacePlacementInfo A_placement_info[8] =
{
    { .channel_id = 6, .tile_offset = 0 },
    { .channel_id = 7, .tile_offset = 0 },
    { .channel_id = 6, .tile_offset = 1 },
    { .channel_id = 7, .tile_offset = 1 },
    { .channel_id = 6, .tile_offset = 2 },
    { .channel_id = 7, .tile_offset = 2 },
    { .channel_id = 6, .tile_offset = 3 },
    { .channel_id = 7, .tile_offset = 3 }
};
uint32_t B_base;
uint32_t B_height;
uint32_t B_width;
uint32_t B_tile_height;
uint32_t B_tile_width;
const DacePlacementInfo B_placement_info[8] =
{
    { .channel_id = 0, .tile_offset = 0 },
    { .channel_id = 1, .tile_offset = 0 },
    { .channel_id = 0, .tile_offset = 1 },
    { .channel_id = 1, .tile_offset = 1 },
    { .channel_id = 0, .tile_offset = 2 },
    { .channel_id = 1, .tile_offset = 2 },
    { .channel_id = 0, .tile_offset = 3 },
    { .channel_id = 1, .tile_offset = 3 }
};
uint32_t C_base;
uint32_t C_height;
uint32_t C_width;
uint32_t C_tile_height;
uint32_t C_tile_width;
const DacePlacementInfo C_placement_info[64] =
{
    { .channel_id = 0, .tile_offset = 0 },
    { .channel_id = 1, .tile_offset = 0 },
    { .channel_id = 0, .tile_offset = 1 },
    { .channel_id = 1, .tile_offset = 1 },
    { .channel_id = 0, .tile_offset = 2 },
    { .channel_id = 1, .tile_offset = 2 },
    { .channel_id = 0, .tile_offset = 3 },
    { .channel_id = 1, .tile_offset = 3 },
    { .channel_id = 0, .tile_offset = 4 },
    { .channel_id = 1, .tile_offset = 4 },
    { .channel_id = 0, .tile_offset = 5 },
    { .channel_id = 1, .tile_offset = 5 },
    { .channel_id = 0, .tile_offset = 6 },
    { .channel_id = 1, .tile_offset = 6 },
    { .channel_id = 0, .tile_offset = 7 },
    { .channel_id = 1, .tile_offset = 7 },
    { .channel_id = 0, .tile_offset = 8 },
    { .channel_id = 1, .tile_offset = 8 },
    { .channel_id = 0, .tile_offset = 9 },
    { .channel_id = 1, .tile_offset = 9 },
    { .channel_id = 0, .tile_offset = 10 },
    { .channel_id = 1, .tile_offset = 10 },
    { .channel_id = 0, .tile_offset = 11 },
    { .channel_id = 1, .tile_offset = 11 },
    { .channel_id = 0, .tile_offset = 12 },
    { .channel_id = 1, .tile_offset = 12 },
    { .channel_id = 0, .tile_offset = 13 },
    { .channel_id = 1, .tile_offset = 13 },
    { .channel_id = 0, .tile_offset = 14 },
    { .channel_id = 1, .tile_offset = 14 },
    { .channel_id = 0, .tile_offset = 15 },
    { .channel_id = 1, .tile_offset = 15 },
    { .channel_id = 0, .tile_offset = 16 },
    { .channel_id = 1, .tile_offset = 16 },
    { .channel_id = 0, .tile_offset = 17 },
    { .channel_id = 1, .tile_offset = 17 },
    { .channel_id = 0, .tile_offset = 18 },
    { .channel_id = 1, .tile_offset = 18 },
    { .channel_id = 0, .tile_offset = 19 },
    { .channel_id = 1, .tile_offset = 19 },
    { .channel_id = 0, .tile_offset = 20 },
    { .channel_id = 1, .tile_offset = 20 },
    { .channel_id = 0, .tile_offset = 21 },
    { .channel_id = 1, .tile_offset = 21 },
    { .channel_id = 0, .tile_offset = 22 },
    { .channel_id = 1, .tile_offset = 22 },
    { .channel_id = 0, .tile_offset = 23 },
    { .channel_id = 1, .tile_offset = 23 },
    { .channel_id = 0, .tile_offset = 24 },
    { .channel_id = 1, .tile_offset = 24 },
    { .channel_id = 0, .tile_offset = 25 },
    { .channel_id = 1, .tile_offset = 25 },
    { .channel_id = 0, .tile_offset = 26 },
    { .channel_id = 1, .tile_offset = 26 },
    { .channel_id = 0, .tile_offset = 27 },
    { .channel_id = 1, .tile_offset = 27 },
    { .channel_id = 0, .tile_offset = 28 },
    { .channel_id = 1, .tile_offset = 28 },
    { .channel_id = 0, .tile_offset = 29 },
    { .channel_id = 1, .tile_offset = 29 },
    { .channel_id = 0, .tile_offset = 30 },
    { .channel_id = 1, .tile_offset = 30 },
    { .channel_id = 0, .tile_offset = 31 },
    { .channel_id = 1, .tile_offset = 31 }
};


int __dace_init_cuda(struct GEMM_state_t *__state) {
    
    

    // __state->gpu_context = new dace::cuda::Context(2, 1);

    

    return 0;
}

int __dace_exit_cuda(struct GEMM_state_t *__state) {
    
    int __err = 0;
    // delete __state->gpu_context;
    return __err;
}


void gemm_entry_0_0_0(const uint32_t A, const uint32_t B, const uint32_t C) {
    {
        // TEST KERNEL SCOPE
        const uint32_t cluster_id = flex_get_cluster_id();
        {
            for (int i = 0; i < 512; i += 128) {
                for (int j = 0; j < 512; j += 128) {
                    {
                        // TEST DEVICE SCOPE
                        int gi = cluster_id % 2;
                        int gj = cluster_id / 2;
                        if (gj <= 1) {
                            if (gi <= 1) {
                                // Minels: [0, 0], Maxels: [1, 1]
                                // Configure RedMule Here
                                if(flex_is_first_core())
                                {
                                    flex_redmule_config(64, 128, 64);
                                }
                                {
                                    for (int ci = 0; ci < 64; ci += 64) {
                                        for (int cj = 0; cj < 64; cj += 64) {
                                            uint32_t accumulator;
                                            accumulator = 0;
                                            // accumulator = accumulator;
                                            {
                                                for (int bK = 0; bK < 512; bK += 128) {
                                                    uint32_t local_A;
                                                    local_A = 8192;
                                                    uint32_t local_B;
                                                    local_B = 24576;
                                                    // local_A = local_A;
                                                    // copy_memory: A -> local_A, [64, 128], [512, 1], [128, 1], A + (((bK + (32768 * ci)) + (32768 * gi)) + (512 * i)), local_A
                                                    // is_sync = True
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_get_core_id() == (0 % (ARCH_NUM_CORE_PER_CLUSTER - 1)) + 1)
                                                    {
                                                        const uint32_t tile_width = A_tile_width;
                                                        const uint32_t tile_height = A_tile_height;
                                                        const uint32_t row_start_offset = ((A - A_base) / 2) / A_width;
                                                        const uint32_t col_start_offset = ((A - A_base) / 2) % A_width;
                                                        const uint32_t col_start_temp = bK + col_start_offset;
                                                        const uint32_t col_start = col_start_temp % A_width;
                                                        const uint32_t row_start = (((64 * ci) + (64 * gi)) + i) + row_start_offset + col_start_temp / A_width;
                                                        const uint32_t tile_row_index = row_start/tile_height;
                                                        const uint32_t tile_col_index = col_start/tile_width;
                                                        const uint32_t tile_row_offset = row_start%tile_height;
                                                        const uint32_t tile_col_offset = col_start%tile_width;
                                                        const uint32_t tile_index = tile_row_index*1 + tile_col_index;
                                                        const uint32_t channel_id = A_placement_info[tile_index].channel_id;
                                                        const uint32_t num_blocks_per_tile = (tile_height/64) * (tile_width/128);
                                                        const uint32_t num_blocks_in_previous_tiles_in_channel = A_placement_info[tile_index].tile_offset * num_blocks_per_tile;
                                                        const uint32_t block_row_index = tile_row_offset/64;
                                                        const uint32_t block_col_index = tile_col_offset/128;
                                                        const uint32_t block_index = block_row_index * (tile_width/128) + block_col_index;
                                                        const uint32_t total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                        const uint64_t block_addr = A_base + channel_id * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 128 * 2;
                                                        flex_dma_async_1d(local(local_A), hbm_addr(block_addr), 64*128*2);
                                                        flex_dma_async_wait_all();
                                                    }
                                                    // local_B = local_B;
                                                    // copy_memory: B -> local_B, [128, 64], [512, 1], [64, 1], B + ((((512 * bK) + (64 * cj)) + (64 * gj)) + j), local_B
                                                    // is_sync = True
                                                    // SoftHier_HBM -> SoftHier_TCDM 2D
                                                    if(flex_get_core_id() == (1 % (ARCH_NUM_CORE_PER_CLUSTER - 1)) + 1)
                                                    {
                                                        const uint32_t tile_width = B_tile_width;
                                                        const uint32_t tile_height = B_tile_height;
                                                        const uint32_t row_start_offset = ((B - B_base) / 2) / B_width;
                                                        const uint32_t col_start_offset = ((B - B_base) / 2) % B_width;
                                                        const uint32_t col_start_temp = (((64 * cj) + (64 * gj)) + j) + col_start_offset;
                                                        const uint32_t col_start = col_start_temp % B_width;
                                                        const uint32_t row_start = bK + row_start_offset + col_start_temp / B_width;
                                                        const uint32_t tile_row_index = row_start/tile_height;
                                                        const uint32_t tile_col_index = col_start/tile_width;
                                                        const uint32_t tile_row_offset = row_start%tile_height;
                                                        const uint32_t tile_col_offset = col_start%tile_width;
                                                        const uint32_t tile_index = tile_row_index*8 + tile_col_index;
                                                        const uint32_t channel_id = B_placement_info[tile_index].channel_id;
                                                        const uint32_t num_blocks_per_tile = (tile_height/128) * (tile_width/64);
                                                        const uint32_t num_blocks_in_previous_tiles_in_channel = B_placement_info[tile_index].tile_offset * num_blocks_per_tile;
                                                        const uint32_t block_row_index = tile_row_offset/128;
                                                        const uint32_t block_col_index = tile_col_offset/64;
                                                        const uint32_t block_index = block_row_index * (tile_width/64) + block_col_index;
                                                        const uint32_t total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                        const uint64_t block_addr = B_base + channel_id * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 128 * 64 * 2;
                                                        flex_dma_async_1d(local(local_B), hbm_addr(block_addr), 128*64*2);
                                                        flex_dma_async_wait_all();
                                                    }
                                                    flex_intra_cluster_sync();
                                                    if (flex_is_first_core())
                                                    {
                                                        uint32_t _in_local_a = local_A;
                                                        uint32_t _in_local_b = local_B;
                                                        uint32_t _in_accumulator = accumulator;

                                                        ///////////////////
                                                        flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_NONE_16);
                                                        flex_redmule_wait();
                                                        ///////////////////

                                                    }
                                                    flex_intra_cluster_sync();
                                                }
                                            }
                                            // accumulator = accumulator;
                                            // copy_memory: accumulator -> C, [64, 64], [64, 1], [512, 1], accumulator, C + ((((((32768 * ci) + (64 * cj)) + (32768 * gi)) + (64 * gj)) + (512 * i)) + j)
                                            // is_sync = True
                                            // SoftHier_TCDM -> SoftHier_HBM
                                            if(flex_get_core_id() == (2 % (ARCH_NUM_CORE_PER_CLUSTER - 1)) + 1)
                                            {
                                                const uint32_t tile_width = C_tile_width;
                                                const uint32_t tile_height = C_tile_height;
                                                const uint32_t row_start_offset = ((C - C_base) / 2) / C_width;
                                                const uint32_t col_start_offset = ((C - C_base) / 2) % C_width;
                                                const uint32_t col_start_temp = (((64 * cj) + (64 * gj)) + j) + col_start_offset;
                                                const uint32_t col_start = col_start_temp % C_width;
                                                const uint32_t row_start = (((64 * ci) + (64 * gi)) + i) + row_start_offset + col_start_temp / C_width;
                                                const uint32_t tile_row_index = row_start/tile_height;
                                                const uint32_t tile_col_index = col_start/tile_width;
                                                const uint32_t tile_row_offset = row_start%tile_height;
                                                const uint32_t tile_col_offset = col_start%tile_width;
                                                const uint32_t tile_index = tile_row_index*8 + tile_col_index;
                                                const uint32_t channel_id = C_placement_info[tile_index].channel_id;
                                                const uint32_t num_blocks_per_tile = (tile_height/64) * (tile_width/64);
                                                const uint32_t num_blocks_in_previous_tiles_in_channel = C_placement_info[tile_index].tile_offset * num_blocks_per_tile;
                                                const uint32_t block_row_index = tile_row_offset/64;
                                                const uint32_t block_col_index = tile_col_offset/64;
                                                const uint32_t block_index = block_row_index * (tile_width/64) + block_col_index;
                                                const uint32_t total_block_index = num_blocks_in_previous_tiles_in_channel + block_index;
                                                const uint64_t block_addr = C_base + channel_id * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE + total_block_index * 64 * 64 * 2;
                                                flex_dma_async_1d(hbm_addr(block_addr), local(accumulator), 64*64*2);
                                                flex_dma_async_wait_all();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    flex_intra_cluster_sync();
                    // Finished deivelevel scope
                }
            }
        }
    }
}


void main(GEMM_state_t *__state, uint32_t A, uint32_t B, uint32_t C);
void main(GEMM_state_t *__state, uint32_t A, uint32_t B, uint32_t C)
{
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    A = ((uint32_t *)(hbm_addr(0)))[0];
    B = ((uint32_t *)(hbm_addr(4)))[0];
    C = ((uint32_t *)(hbm_addr(8)))[0];
    A_base = A;
    B_base = B;
    C_base = C;
    A_height = 512;
    A_width = 512;
    A_tile_height = 512/8;
    A_tile_width = 512/1;
    B_height = 512;
    B_width = 512;
    B_tile_height = 512/1;
    B_tile_width = 512/8;
    C_height = 512;
    C_width = 512;
    C_tile_height = 512/8;
    C_tile_width = 512/8;

    // if (flex_is_first_core() && (flex_get_cluster_id()==0))
    // {
        // printf("A: %x\n", A);
        // printf("B: %x\n", B);
        // printf("C: %x\n", C);
        // printf("K: %x\n", K);
        // printf("M: %x\n", M);
        // printf("N: %x\n", N);
    // }
    // if (flex_is_first_core() && (flex_get_cluster_id()==0))
    // {
        // printf("%x\n", ((uint32_t *)(hbm_addr(A)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(B)))[0]);
        // printf("%x\n", ((uint32_t *)(hbm_addr(C)))[0]);
    // }
    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    flex_timer_start();
    gemm_entry_0_0_0(A, B, C);
    flex_global_barrier_xy();
    flex_timer_end();
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
    if (flex_is_dm_core() && (flex_get_cluster_id() == 0))
    {flex_dump_open();
        flex_dump_hbm(A, A_tile_width * A_tile_height);
        flex_dump_close();
    }
    flex_intra_cluster_sync();

    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return;
}


import dace
import os
import numpy as np
import subprocess
import re

from dace.soft_hier import generate_arg_cfg, make_preload_elf_hbm_interleaved_new, InterleaveHandler
from dace.soft_hier import _my_gen_baseline_matmul_sdfg

# Configuration
GVSOC_PATH = "/home/primrose/Work/SoftHier/gvsoc"
THREAD_GROUP_DIMS = (2, 2)
HBM_ADDR_BASE = 0xc0000000
HBM_ADDR_SPACE = 0x20000000
TCDM_SIZE = 0x04000000 // 4  # Total / (2x2)
DTYPE_INPUT = np.uint16
DTYPE_OUTPUT = np.uint16

def setup_environment():
    """Setup paths and environment"""
    os.environ["GVSOC_INSTALL_PATH"] = GVSOC_PATH
    os.environ["GVSOC_DIR"] = GVSOC_PATH
    os.environ["GVSOC_PATH"] = GVSOC_PATH
    os.environ["PATH"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin:{os.environ['PATH']}"
    os.environ["SHCC"] = f"{GVSOC_PATH}/third_party/toolchain/install/bin/riscv32-unknown-elf-gcc"
    os.environ["CPLUS_INCLUDE_PATH"] = f"/home/primrose/Work/SoftHier/softhierdace/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"
    os.environ["C_INCLUDE_PATH"] = f"/home/primrose/Work/SoftHier/softhierdace/dace/runtime/include/dace/soft_hier/runtime/include:{os.environ.get('C_INCLUDE_PATH','')}"
    os.environ["SOFTHIER_INSTALL_PATH"] = f"{GVSOC_PATH}/soft_hier/flex_cluster_sdk/runtime/"
    os.environ["CCACHE_DIR"] = f"/home/primrose/.ccache"

def setup_architecture():
    """Generate and compile architecture"""
    generate_arg_cfg(
        cluster_tcdm_size=hex(TCDM_SIZE*4),
        num_cluster_x=4, num_cluster_y=4,
        redmule_ce_height=64, redmule_ce_width=32, redmule_ce_pipe=1,
        hbm_start_base=hex(HBM_ADDR_BASE),
        hbm_node_addr_space=hex(HBM_ADDR_SPACE),
        hbm_placement="4,0,0,4", num_node_per_ctrl=1, noc_link_width=4096
    )
    
    subprocess.run([
        "bash", "-c", 
        f"cd {GVSOC_PATH} && source sourceme.sh && export cfg={os.path.dirname(__file__)}/generated_arch.py && make hw"
    ], check=True)

def create_test_data(M, N, K, hwM, hwN, hwK):
    """Create test matrices and handlers"""
    A = np.ones((M, K), dtype=DTYPE_INPUT)
    B = np.ones((K, N), dtype=DTYPE_INPUT)
    C = np.zeros((M, N), dtype=DTYPE_OUTPUT)
    
    # Setup handlers
    A_handler = InterleaveHandler(A, (hwM, hwK), THREAD_GROUP_DIMS)
    A_handler.split_horizental()
    A_handler.place_to_range((0, 3, 1))
    
    B_handler = InterleaveHandler(B, (hwK, hwN), THREAD_GROUP_DIMS)
    B_handler.split_vertical()
    B_handler.place_to_range((0, 3, 1))  # 0 to dim_y-1
    
    C_handler = InterleaveHandler(C, (hwM, hwN), THREAD_GROUP_DIMS)
    C_handler.split_to_blocks()
    C_handler.place_to_range((0, 3, 1))
    
    return A, B, C, A_handler, B_handler, C_handler

def step1_np_to_hbm(handlers):
    """Step 1: Move NumPy to HBM"""
    A_handler, B_handler, C_handler = handlers
    make_preload_elf_hbm_interleaved_new(
        "output.elf", handlers, 
        KMN=[A_handler.array.shape[1], A_handler.array.shape[0], B_handler.array.shape[1]],
        hbm_node_addr_base=HBM_ADDR_BASE, hbm_node_addr_space=HBM_ADDR_SPACE
    )
    print("A_handler")
    A_handler.print_info()
    print("B_handler")
    B_handler.print_info()
    print("C_handler")
    C_handler.print_info()
    raise Exception("HALT")

def step2_run_softhier(A, B, C, M, N, K, hwM, hwN, hwK, handlers):
    """Step 2: Run SoftHier kernel"""
    A_handler, B_handler, C_handler = handlers
    
    sdfg = _my_gen_baseline_matmul_sdfg(
        hardware_matmul_mnk=(hwM, hwN, hwK),
        global_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
        thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
        thread_group_dims=THREAD_GROUP_DIMS,
        hbm_split_scheme=[h.split_scheme for h in handlers],
        hbm_placement_scheme=[h.placement_scheme for h in handlers],
        is_hbm_interleaved=True,
        input_float=dace.uint16, output_float=dace.uint16,
        coarsening_factor=1,
        mmad_tasklet_str="flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_NONE_16);",
        GEMM_shape=(M, N, K),
    )
    
    sdfg.validate()
    compiled_sdfg = sdfg.compile()
    compiled_sdfg(A=A, B=B, C=C, M=M, N=N, K=K)
    
    # Parse timing
    try:
        with open("./log", "r") as f:
            for line in f:
                match = re.search(r"Execution period is (\d+) ns", line)
                if match:
                    return int(match.group(1))
    except FileNotFoundError:
        pass
    return None

def step3_hbm_to_np(C):
    """Step 3: Move HBM to NumPy (TODO: implement)"""
    print("TODO: Implement HBM to NP copy")
    return True

def step4_run_numpy(A, B):
    """Step 4: Run NumPy reference"""
    return np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(DTYPE_OUTPUT)

def step5_compare(C_expected, C_softhier, tolerance=1e-3):
    """Step 5: Compare results"""
    diff = np.abs(C_expected.astype(np.float32) - C_softhier.astype(np.float32))
    max_diff = np.max(diff)
    matches = max_diff < tolerance
    return {"matches": matches, "max_diff": max_diff}

def run_test(M, N, K, hwM, hwN, hwK):
    """Run complete test pipeline"""
    print(f"Testing GEMM({M}, {N}, {K}) with HW({hwM}, {hwN}, {hwK})")
    
    # Create data
    A, B, C, *handlers = create_test_data(M, N, K, hwM, hwN, hwK)
    
    # Run pipeline
    step1_np_to_hbm(handlers)
    timing = step2_run_softhier(A, B, C, M, N, K, hwM, hwN, hwK, handlers)
    step3_hbm_to_np(C)
    C_expected = step4_run_numpy(A, B)
    comparison = step5_compare(C_expected, C)
    
    print(f"Result: {comparison['matches']}, Max diff: {comparison['max_diff']:.6f}")
    if timing:
        print(f"Timing: {timing} ns")
    
    return comparison["matches"]

def main():
    setup_environment()
    setup_architecture()
    
    # Run test
    success = run_test(512, 512, 512, 64, 64, 128)
    print(f"Test {'PASSED' if success else 'FAILED'}")

if __name__ == "__main__":
    main()
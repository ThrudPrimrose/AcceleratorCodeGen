import dace
import os
import numpy as np
import subprocess
import sys
import re
import time

from dace.transformation.soft_hier import SystolocTransformer, SystolicTransformer, CannonTransformer, SplitHBMLoad, SystolicSplitStore, SummaTransformer
from dace.soft_hier import generate_arg_cfg, make_preload_elf, make_preload_elf_hbm_interleaved_new, generate_tiling, InterleaveHandler
from dace.soft_hier import _my_gen_systolic_matmul_sdfg, _my_gen_BSP_matmul_sdfg, _my_gen_summa_matmul_sdfg, _my_gen_baseline_matmul_sdfg
from dace.soft_hier import generate_systolic_BSP, generate_summa_BSP, generate_cannon_BSP, generate_summa_systolic_BSP, generate_systolic_summa_BSP, generate_multistream_BSP

N = dace.symbol("N")

def vadd_numpy(A, B, C):
    C = A + B

@dace.program
def vadd_dace(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N]):
    for i in dace.map[0:N:1]:
        C[i] = A[i] + B[i]

vadd_sdfg = vadd_dace.to_sdfg()


class SoftHierConfig:
    """Configuration class for SoftHier parameters"""
    def __init__(self):
        self.hbm_node_addr_space = 0x20000000
        self.hbm_node_addr_base = 0xc0000000
        self.thread_group_dims = (4, 4)
        self.hbm_placement = "4,0,0,4"
        self.hbm_node_per_ctrl = 1
        self.total_tcdm_size = 0x08000000
        self.redmule_h = 64
        self.redmule_w = 32
        self.redmule_ce_pipe = 1
        
        # Data types
        self.dace_data_type_input = dace.uint16
        self.dace_data_type_output = dace.uint16
        self.numpy_data_type_input = np.uint16
        self.numpy_data_type_output = np.uint16
        
        # Paths
        self.gvsoc_path = "/home/primrose/Work/SoftHier/gvsoc"
        self.ccache_path = "/usr/bin"
        self.python_script_path = os.path.dirname(os.path.realpath(__file__))
        self.temp_run_dir = str(os.environ.get("SOFTHIER_TEMP_RUN_PATH", self.python_script_path))
        
        # Derived properties
        self.dim_x = self.thread_group_dims[0]
        self.dim_y = self.thread_group_dims[1]
        self.tcdm_size = self.total_tcdm_size // (self.dim_x * self.dim_y)
        self.elem_size_input = self.numpy_data_type_input().itemsize
        self.elem_size_output = self.numpy_data_type_output().itemsize
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables"""
        os.environ["GVSOC_INSTALL_PATH"] = self.gvsoc_path
        os.environ["GVSOC_DIR"] = self.gvsoc_path
        os.environ["SOFTHIER_INSTALL_PATH"] = f"{self.gvsoc_path}/soft_hier/flex_cluster_sdk/runtime/"
        os.environ["CCACHE_DIR"] = f"/home/primrose/.ccache"
        os.environ["PATH"] = f"{self.gvsoc_path}/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin:{os.environ['PATH']}"
        os.environ["SHCC"] = f"{self.gvsoc_path}/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin/riscv32-unknown-elf-gcc"


########################################################################
# Worker function: Runs ONE parameter combination in a temp directory
########################################################################
def run_sdfg_in_tempdir(combo):
    """
    Each call uses the SLOT environment variable set in init_worker.
    Returns a dict of the relevant parameters plus the measured execution_period_ns.
    """
    # Retrieve the SLOT assigned to this worker process
    slot_id = os.environ.get("SLOT", "UNKNOWN")

    (
        M_val,
        N_val,
        K_val,
        hwM,
        hwN,
        hwK,
        thread_group_dims,
        tcdm_size
    ) = combo

    (dim_x, dim_y) = thread_group_dims

    hardware_matmul_mnk = (hwM, hwN, hwK)
    combo_summary = (
        f"SLOT={slot_id}, "
        f"M={M_val}, N={N_val}, K={K_val}, "
        f"hwMNK={hardware_matmul_mnk}"
    )
    log_file = open(log_path, "a")
    print(f"[{time.asctime()}] Starting {combo_summary}", flush=True, file=log_file)
    log_file.close()
    # Redirect stdout and stderr to a log file
    slot_dir = f"{temp_run_dir}/slot_{slot_id}"
    # log_file_path = ""
    execution_period_ns = None
    start_time = time.time()

    tmp_dir = "."
    A_host = np.ones((M_val, K_val), dtype=numpy_data_type_input)
    B_host = np.ones((K_val, N_val), dtype=numpy_data_type_input)
    C_host = np.zeros((M_val, N_val), dtype=numpy_data_type_output)

    A_handler = InterleaveHandler(array=A_host, block_shape=(hwM, hwK), cluster_dims=thread_group_dims)
    A_handler.split_horizental()
    A_handler.place_to_range(place_range=(dim_x+2*dim_y, 2*dim_x+2*dim_y-1, 1))
    split_A = A_handler.split_scheme
    place_A = A_handler.placement_scheme

    B_handler = InterleaveHandler(array=B_host, block_shape=(hwK, hwN), cluster_dims=thread_group_dims)
    B_handler.split_vertical()
    B_handler.place_to_range(place_range=(0, dim_y - 1, 1))
    split_B = B_handler.split_scheme
    place_B = B_handler.placement_scheme

    C_handler = InterleaveHandler(array=C_host, block_shape=(hwM, hwN), cluster_dims=thread_group_dims)
    C_handler.split_to_blocks()
    C_handler.place_to_range(place_range=(0, dim_y - 1, 1))
    split_C = C_handler.split_scheme
    place_C = C_handler.placement_scheme

    make_preload_elf_hbm_interleaved_new(
        "output.elf",
        [A_handler, B_handler, C_handler],
        KMN=[K_val, M_val, N_val],
        hbm_node_addr_base=hbm_node_addr_base,
        hbm_node_addr_space=hbm_node_addr_space
    )

    M = M_val
    N = N_val
    K = K_val

    sdfg = _my_gen_baseline_matmul_sdfg(
        hardware_matmul_mnk=hardware_matmul_mnk,
        global_storage=dace.dtypes.StorageType.SoftHier_HBM,
        local_storage=dace.dtypes.StorageType.SoftHier_TCDM,
        device_schedule=dace.dtypes.ScheduleType.SoftHier_Device,
        thread_group_schedule=dace.dtypes.ScheduleType.SoftHier_Cluster,
        thread_group_dims=thread_group_dims,
        hbm_split_scheme=[split_A, split_B, split_C],
        hbm_placement_scheme=[place_A, place_B, place_C],
        is_hbm_interleaved=True,
        input_float=dace_data_type_input,
        output_float=dace_data_type_output,
        coarsening_factor=1,
        mmad_tasklet_str="flex_redmule_trigger(_in_local_a, _in_local_b, _in_accumulator, REDMULE_NONE_16);",
        GEMM_shape=(M_val, N_val, K_val),
    )

    sdfg.validate()
    sdfg.save("matmul_base.sdfgz")
    compiled_sdfg = sdfg.compile()
    compiled_sdfg(A=A_host, B=B_host, C=C_host,
                    M=M_val, N=N_val, K=K_val)
    # flush the stdout/stderr
    sys.stdout.flush()
    sys.stderr.flush()
    # flush the log file
    #os.system(f"cp -rf {tmp_dir} {python_script_path}")

    # Parse the log file for the performance counter
    with open("./log", "r") as log_file:
        for line in log_file:
            match = re.search(r"\[Performance Counter\]: Execution period is (\d+) ns", line)
            if match:
                execution_period_ns = int(match.group(1))
                break
            else:
                execution_period_ns = None


    end_time = time.time()

    

    # Print a short summary
    duration_s = end_time - start_time
    log_file = open(log_path, "a")
    print(f"[{time.asctime()}] Completed {combo_summary} in {duration_s:.2f} seconds; "
          f"period={execution_period_ns} ns", flush=True, file=log_file)
    log_file.close()

    # Return all relevant info in a dictionary
    if not execution_period_ns:
        return None
    else:
        return {
            "thread_group_dims": thread_group_dims,
            "M": M_val,
            "N": N_val,
            "K": K_val,
            "hwM": hwM,
            "hwN": hwN,
            "hwK": hwK,
            "execution_period_ns": execution_period_ns
        }

########################################################################
# Main: build parameter sweeps, then run them in parallel
#       Each worker process has a unique SLOT env variable
########################################################################

import subprocess
def get_path(binary_name):
    # Run 'whereis' and capture stdout
    result = subprocess.run(["whereis", binary_name], capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split()
    # whereis output looks like: "gvsoc: /users/ashen/dace4softhier/gvsoc"
    if len(parts) > 1:
        return parts[1]
    return None

hbm_node_addr_space = 0x20000000
hbm_node_addr_base  = 0xc0000000
script_start_time=time.time()
SLURM_JOB_ID = str(os.environ.get("SLURM_JOB_ID"))
max_procs = int(os.environ.get("MAX_PROCS", 4))
gvsoc_path = "/home/primrose/Work/SoftHier/gvsoc"
ccache_path = "/usr/bin"
python_script_path = os.path.dirname(os.path.realpath(__file__))
temp_run_dir = str(os.environ.get("SOFTHIER_TEMP_RUN_PATH", python_script_path))
log_path = f"{python_script_path}/sweeplog_{SLURM_JOB_ID}.txt"
dace_data_type_input = dace.uint16
dace_data_type_output = dace.uint16
numpy_data_type_input = np.uint16
numpy_data_type_output = np.uint16
elem_size_input = numpy_data_type_input().itemsize
elem_size_output = numpy_data_type_output().itemsize
os.environ["GVSOC_INSTALL_PATH"] = gvsoc_path
os.environ["GVSOC_DIR"] = gvsoc_path
os.environ["SOFTHIER_INSTALL_PATH"] = f"{gvsoc_path}/soft_hier/flex_cluster_sdk/runtime/"
os.environ["CCACHE_DIR"] = f"/home/primrose/.ccache"
os.environ["PATH"] = f"/home/primrose/Work/SoftHier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin:{os.environ["PATH"]}"
os.environ["SHCC"] = f"/home/primrose/Work/SoftHier/gvsoc/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7/bin/riscv32-unknown-elf-gcc"

print(os.environ["SHCC"])
print(os.environ["PATH"])

if __name__ == "__main__":
    # Example parameter lists
    thread_group_dims = (4, 4)
    hbm_placement = "4,0,0,4"
    hbm_node_per_ctrl = 1
    total_tcdm_size = 0x08000000
    dim_x = thread_group_dims[0]
    dim_y = thread_group_dims[1]
    redmule_h = 64
    redmule_w = 32
    redmule_ce_pipe = 1
    tcdm_size = total_tcdm_size // (dim_x * dim_y)
    csv_filename = f"Dim_{dim_x}x{dim_y}_RedMule_{redmule_h}x{redmule_w}_TCDM_{hex(tcdm_size)}.csv"
    os.makedirs(f"{python_script_path}/logs", exist_ok=True)
    
    # Generate the architecture configuration
    generate_arg_cfg(
        cluster_tcdm_size=hex(tcdm_size*2),
        num_cluster_x=dim_x,
        num_cluster_y=dim_y,
        redmule_ce_height=redmule_h,
        redmule_ce_width=redmule_w,
        redmule_ce_pipe=redmule_ce_pipe,
        hbm_start_base=hex(hbm_node_addr_base),
        hbm_node_addr_space=hex(hbm_node_addr_space),
        hbm_placement=hbm_placement,
        num_node_per_ctrl=hbm_node_per_ctrl,
        noc_link_width=4096
    )

    M_list = [512]
    N_K_list = [(512, 512)]

    
    # [Optional] Re-compile the generated SoftHier architecture
    subprocess.run([
         "bash", "-c", 
         f"cd {gvsoc_path} && source sourceme.sh && export cfg={python_script_path}/generated_arch.py && make hw"
     ], check=True)


    combo = (512,512,512,64,64,128, thread_group_dims, tcdm_size)

    run_sdfg_in_tempdir(combo)
    
    
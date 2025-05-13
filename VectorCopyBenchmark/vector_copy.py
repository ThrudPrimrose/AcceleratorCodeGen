import dace
import shutil
import os
import csv
import numpy as np

NUM_AI_CORES = 20

def vector_add(name, vector_size: int, frag_size: int):
    sdfg = dace.SDFG(name)
    wstate = sdfg.add_state("warmup")
    mstate = sdfg.add_state("main")


    a_host = sdfg.add_array("A", (vector_size, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=False)
    a_dev = sdfg.add_array("ascend_A", (vector_size, ), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    b_host = sdfg.add_array("B", (vector_size, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=False)
    b_dev = sdfg.add_array("ascend_B", (vector_size, ), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    c_host = sdfg.add_array("C", (vector_size, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=False)
    c_dev = sdfg.add_array("ascend_C", (vector_size, ), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)

    for i, state in enumerate([wstate, mstate]):
        ahc = state.add_access("A")
        adc = state.add_access("ascend_A")
        bhc = state.add_access("B")
        bdc = state.add_access("ascend_B")
        chc = state.add_access("C")
        cdc = state.add_access("ascend_C")

        frag_a = sdfg.add_array(f"frag{i}_A", (frag_size, ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
        frag_b = sdfg.add_array(f"frag{i}_B", (frag_size, ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
        frag_c = sdfg.add_array(f"frag{i}_C", (frag_size, ), dace.float16, dace.dtypes.StorageType.Ascend_VECOUT, transient=True)


        dev_entry, dev_exit = state.add_map(name="copy_map_outer", ndrange={"i": dace.subsets.Range([(0, vector_size-1, frag_size*NUM_AI_CORES)])}, schedule=dace.dtypes.ScheduleType.Ascend_Device)
        tblock_entry, tblock_exit = state.add_map(name="copy_map_inner", ndrange={"ii": dace.subsets.Range([(0, frag_size*NUM_AI_CORES-1, frag_size)])}, schedule=dace.dtypes.ScheduleType.Ascend_AiCoreGroup)
        if state == mstate:
            dev_entry.instrument = dace.InstrumentationType.Timer

        glb_to_vecin1 = state.add_access(f"frag{i}_A")
        glb_to_vecin2 = state.add_access(f"frag{i}_B")
        libnode = state.add_access(f"frag{i}_C")

        for ahc, adc, glb_to_vecin, n in [(ahc, adc, glb_to_vecin1, "A"), (bhc, bdc, glb_to_vecin2, "B")]:
            state.add_edge(ahc, None, adc, None, dace.memlet.Memlet(f"{n}[0:{vector_size}]"))
            state.add_edge(adc, None, dev_entry, "IN_" + n, dace.memlet.Memlet(f"ascend_{n}[0:{vector_size}]"))
            state.add_edge(dev_entry, "OUT_" + n, tblock_entry, "IN_" + n, dace.memlet.Memlet(f"ascend_{n}[0:{vector_size}]"))
            state.add_edge(tblock_entry, "OUT_" + n, glb_to_vecin, None, dace.memlet.Memlet(f"ascend_{n}[i + ii:i + ii + {frag_size}]"))

        tasklet = state.add_tasklet(name="Add",
                                    inputs={"IN_frag_A", "IN_frag_B"},
                                    outputs={"OUT_frag_C"},
                                    code=f"Add(OUT_frag_C, IN_frag_A, IN_frag_B, {frag_size})")
        state.add_edge(glb_to_vecin1, None, tasklet, "IN_frag_A", dace.memlet.Memlet(f"frag{i}_A[i + ii:i + ii + {frag_size}]"))
        state.add_edge(glb_to_vecin2, None, tasklet, "IN_frag_B", dace.memlet.Memlet(f"frag{i}_B[i + ii:i + ii + {frag_size}]"))
        state.add_edge(tasklet, "OUT_frag_C", libnode, None, dace.memlet.Memlet(f"frag{i}_C[i + ii:i + ii + {frag_size}]"))


        state.add_edge(libnode, None, tblock_exit, "IN_C", dace.memlet.Memlet(f"ascend_C[i + ii:i + ii + {frag_size}]"))
        state.add_edge(tblock_exit, "OUT_C", dev_exit, "IN_C", dace.memlet.Memlet(f"ascend_C[i + ii:i + ii + {frag_size}]"))
        state.add_edge(dev_exit, "OUT_C", cdc, None, dace.memlet.Memlet(f"ascend_C[0:{vector_size}]"))
        state.add_edge(cdc, None, chc, None, dace.memlet.Memlet(f"C[0:{vector_size}]"))

        for n in [dev_entry, tblock_entry]:
            n.add_in_connector("IN_A")
            n.add_out_connector("OUT_A")
            n.add_in_connector("IN_B")
            n.add_out_connector("OUT_B")
        for n in [dev_exit, tblock_exit]:
            n.add_in_connector("IN_C")
            n.add_out_connector("OUT_C")

    sdfg.add_edge(wstate, mstate, dace.InterstateEdge(None))

    return sdfg

cache_dir = ".dacecache"  # The directory to clean
csv_path = "kernel_runtimes_no_sync_pipe_all_2.csv"  # Path to the CSV file
N = 30  # Number of times to run each kernel combination

# Load existing measurements if the CSV exists
existing_measurements = set()
if os.path.exists(csv_path):
    with open(csv_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # Skip header row
        for row in reader:
            vector_size, frag_size = int(row[0]), int(row[1])
            existing_measurements.add((vector_size, frag_size))

def warmup():
    fsize = 32
    vsize=fsize*1000*NUM_AI_CORES
    sdfg = vector_add(name="warmup", vector_size=vsize, frag_size=fsize)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    A = np.full(vsize, 1, dtype=np.float16)  # Array A initialized to 1
    B = np.full(vsize, 2, dtype=np.float16)  # Array B initialized to 2
    C = np.full(vsize, 0, dtype=np.float16)
    sdfg(A=A, B=B, C=C)
    #if os.path.exists(cache_dir):
    #    shutil.rmtree(cache_dir)
    del sdfg
warmup()

exit()
# Loop through the vector sizes (from 8192 * 1 to 8192 * 1024 in increments of 32)
# Warm up
j = 0

for vector_multiplier in range(24, 33, 1):
    vector_size = 2**vector_multiplier

    A = np.full(vector_size, 1, dtype=np.float16)  # Array A initialized to 1
    B = np.full(vector_size, 2, dtype=np.float16)  # Array B initialized to 2
    C = np.full(vector_size, 0, dtype=np.float16)

    # Loop through the fragment sizes (from 32 to 1024 in increments of 32)
    #for frag_power in range(5, 13, 1):
    for frag_power in range(7, 16, 1):
        j += 1
        frag_size = 2 ** frag_power
        # Skip if this combination has already been measured
        if (vector_size, frag_size) in existing_measurements:
            print(f"Skipping vector_size={vector_size}, frag_size={frag_size} (already measured)")
            continue
        if frag_size*NUM_AI_CORES > vector_size:
            print(f"Skipping vector_size={vector_size}, frag_size={frag_size} (vector not long enough)")
            continue
        if vector_size % frag_size*3 != 0:
            print(f"Skipping vector_size={vector_size}, frag_size={frag_size} (remainder loops necessary)")
            continue

        runtimes = []
        valid = True
        for i in range(N):
            try:
                # Clean the .dacecache directory
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"Cleaned cache directory: {cache_dir}")

                # Call the SDFG function with the current vector_size and frag_size
                print(f"Running sdfg with vector_size={vector_size} and frag_size={frag_size} (run {i+1}/{N})")

                sdfg = vector_add(name=f"test_{j}", vector_size=vector_size, frag_size=frag_size)
                sdfg(A=A, B=B, C=C)
                report = sdfg.get_latest_report()
                kernel_time = report.events[0].duration * 1e-3  # Assuming the first event is the kernel time
                #print(report, kernel_time)
                del sdfg
                if not np.allclose(C, 3.0):
                    print(C)
                    valid = False
                    report = sdfg.get_latest_report()
                    kernel_time = report.events[0].duration * 1e-3  # Assuming the first event is the kernel time
                    print(f"Kernel time for vector_size={vector_size}, frag_size={frag_size}, run {i+1}/{N}: {kernel_time} ms")
                    count_3 = np.count_nonzero(C == 3.0)
                    count_0 = np.count_nonzero(C == 0.0)

                    # Calculate percentages
                    total_elements = C.size
                    percent_3 = (count_3 / total_elements) * 100
                    percent_0 = (count_0 / total_elements) * 100

                    # Print results
                    print(f"Percentage of elements that are 3.0 {count_3}/{total_elements}: {percent_3:.2f}%")
                    print(f"Percentage of elements that are 0.0 {count_0}/{total_elements}: {percent_0:.2f}%")
                    break

                # Retrieve and store the duration of the kernel
                print(f"Kernel time for vector_size={vector_size}, frag_size={frag_size}, run {i+1}/{N}: {kernel_time} ms")
                runtimes.append(kernel_time)
            except Exception as e:
                print(f"Kernel time for vector_size={vector_size}, frag_size={frag_size}, runs failed")
                print(f"Exception {e}")
                valid = False
                break

        if len(runtimes) < N:
            runtimes.extend([-1] * (N - len(runtimes)))

        # Save the runtimes to the CSV file
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
                writer.writerow(["vector_size", f"frag_size", "runtimes", "valid"])  # Write header row if file is new
            writer.writerow([vector_size, frag_size, runtimes, valid])

        print(f"Saved runtimes for vector_size={vector_size}, frag_size={frag_size} to {csv_path}")
import dace
import numpy as np

def gpu_one():
    sdfg = dace.SDFG("gpu_test_1")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (512, ), dace.float32, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("ascend_A", (512, ), dace.float32, dace.dtypes.StorageType.GPU_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet("A[0:512]"))
    sdfg.save("ascend_one.sdfgz")
    return sdfg

def one():
    sdfg = dace.SDFG("ascendc_test_1")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (512, ), dace.float32, dace.dtypes.StorageType.CPU_Heap, transient=False)
    a_dev = sdfg.add_array("ascend_A", (512, ), dace.float32, dace.dtypes.StorageType.Ascend_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet("A[0:512]"))
    sdfg.save("ascend_one.sdfgz")
    return sdfg


def two():
    N = dace.symbol("N")
    sdfg = dace.SDFG("ascendc_test_1")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (N, ), dace.float32, dace.dtypes.StorageType.CPU_Heap, transient=False)
    a_dev = sdfg.add_array("ascend_A", (N, ), dace.float32, dace.dtypes.StorageType.Ascend_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet("A[0:N]"))

    map_entry, map_exit = state.add_map(name="vector_copy", ndrange={"i": dace.subsets.Range([0, N, 1]) },
                                        schedule=dace.dtypes.ScheduleType.Ascend_device, unroll=False)




    sdfg.save("ascend_two.sdfgz")
    return sdfg


s = one()
c = s.compile()
array = np.random.rand(512).astype(np.float32)
c(A=array)
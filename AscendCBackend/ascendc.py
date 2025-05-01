import dace

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
    a_host = sdfg.add_array("A", (512, ), dace.float32, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("ascend_A", (512, ), dace.float32, dace.dtypes.StorageType.Ascend_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet("A[0:512]"))
    sdfg.save("ascend_one.sdfgz")
    return sdfg


s = one()
s.compile()

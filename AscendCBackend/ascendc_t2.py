import dace

def two():
    sdfg = dace.SDFG("ascendc_test_2")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (256*32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("ascend_A", (256*32, ), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    b_host = sdfg.add_array("B", (256*32, ), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    b_dev = sdfg.add_array("ascend_B", (256*32, ), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    bhc = state.add_access("B")
    bdc = state.add_access("ascend_B")
    frag_a = sdfg.add_array("frag_A", (256, ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
    frag_b = sdfg.add_array("frag_B", (256, ), dace.float16, dace.dtypes.StorageType.Ascend_VECOUT, transient=True)

    dev_entry, dev_exit = state.add_map(name="copy_map_outer", ndrange={"i": dace.subsets.Range([(0, 256*32-1, 256*32)])}, schedule=dace.dtypes.ScheduleType.Ascend_Device)
    tblock_entry, tblock_exit = state.add_map(name="copy_map_inner", ndrange={"ii": dace.subsets.Range([(0, 256*32-1, 256)])}, schedule=dace.dtypes.ScheduleType.Ascend_AiCoreGroup)

    glb_to_vecin = state.add_access("frag_A")
    libnode = state.add_access("frag_B")

    state.add_edge(ahc, None, adc, None, dace.memlet.Memlet(f"A[0:{256*32}]"))
    state.add_edge(adc, None, dev_entry, "IN_A", dace.memlet.Memlet(f"ascend_A[0:{256*32}]"))
    state.add_edge(dev_entry, "OUT_A", tblock_entry, "IN_A", dace.memlet.Memlet(f"ascend_A[0:{256*32}]"))
    state.add_edge(tblock_entry, "OUT_A", glb_to_vecin, None, dace.memlet.Memlet(f"ascend_A[i + ii:i + ii + 256]"))
    state.add_edge(glb_to_vecin, None, libnode, None, dace.memlet.Memlet(f"frag_A[0:256]"))
    state.add_edge(libnode, None, tblock_exit, "IN_B", dace.memlet.Memlet(f"ascend_B[i + ii:i + ii + 256]"))
    state.add_edge(tblock_exit, "OUT_B", dev_exit, "IN_B", dace.memlet.Memlet(f"ascend_B[i + ii:i + ii + 256]"))
    state.add_edge(dev_exit, "OUT_B", bdc, None, dace.memlet.Memlet(f"ascend_B[0:{256*32}]"))
    state.add_edge(bdc, None, bhc, None, dace.memlet.Memlet(f"B[0:{256*32}]"))

    for n in [dev_entry, tblock_entry]:
        n.add_in_connector("IN_A")
        n.add_out_connector("OUT_A")

    for n in [dev_exit, tblock_exit]:
        n.add_in_connector("IN_B")
        n.add_out_connector("OUT_B")

    sdfg.save("ascend_2.sdfgz")
    return sdfg


s = two()
s.compile()

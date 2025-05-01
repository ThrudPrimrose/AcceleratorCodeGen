import dace

def three():
    sdfg = dace.SDFG("ascendc_test_4")
    state = sdfg.add_state("main")
    a_host = sdfg.add_array("A", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    a_dev = sdfg.add_array("ascend_A", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    b_host = sdfg.add_array("B", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    b_dev = sdfg.add_array("ascend_B", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    c_host = sdfg.add_array("C", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.CPU_Heap, transient=True)
    c_dev = sdfg.add_array("ascend_C", (256*32, 256*32), dace.float16, dace.dtypes.StorageType.Ascend_Global, transient=True)
    ahc = state.add_access("A")
    adc = state.add_access("ascend_A")
    bhc = state.add_access("B")
    bdc = state.add_access("ascend_B")
    chc = state.add_access("C")
    cdc = state.add_access("ascend_C")
    frag_a = sdfg.add_array("frag_A", (1,256 ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
    frag_b = sdfg.add_array("frag_B", (1,256 ), dace.float16, dace.dtypes.StorageType.Ascend_VECIN, transient=True)
    frag_c = sdfg.add_array("frag_C", (1,256 ), dace.float16, dace.dtypes.StorageType.Ascend_VECOUT, transient=True)

    dev_entry, dev_exit = state.add_map(name="copy_map_outer", ndrange={"i": dace.subsets.Range([(0, 256*32-1, 1)]),
                                                                        "j": dace.subsets.Range([(0, 256*32-1, 256*32)]),}, schedule=dace.dtypes.ScheduleType.Ascend_Device)
    tblock_entry, tblock_exit = state.add_map(name="copy_map_inner", ndrange={"ii": dace.subsets.Range([(0, 256*32-1, 1)]),
                                                                              "jj": dace.subsets.Range([(0, 256*32-1, 256)]),}, schedule=dace.dtypes.ScheduleType.Ascend_AiCoreGroup)

    glb_to_vecin1 = state.add_access("frag_A")
    glb_to_vecin2 = state.add_access("frag_B")
    libnode = state.add_access("frag_C")

    for ahc, adc, glb_to_vecin, n in [(ahc, adc, glb_to_vecin1, "A"), (bhc, bdc, glb_to_vecin2, "B")]:
        state.add_edge(ahc, None, adc, None, dace.memlet.Memlet(f"{n}[0:{256*32}, 0:{256*32}]"))
        state.add_edge(adc, None, dev_entry, "IN_" + n, dace.memlet.Memlet(f"ascend_{n}[0:{256*32}, 0:{256*32}]"))
        state.add_edge(dev_entry, "OUT_" + n, tblock_entry, "IN_" + n, dace.memlet.Memlet(f"ascend_{n}[0:{256*32}, 0:{256*32}]"))
        state.add_edge(tblock_entry, "OUT_" + n, glb_to_vecin, None, dace.memlet.Memlet(f"ascend_{n}[i + ii:i + ii, j + jj:j+jj + 256]"))

    #state.add_edge(glb_to_vecin, None, libnode, None, dace.memlet.Memlet(f"frag_A[0:256]"))
    tasklet = state.add_tasklet(name="Add",
                                inputs={"IN_frag_A", "IN_frag_B"},
                                outputs={"OUT_frag_C"},
                                code="Add(OUT_frag_C, IN_frag_A, IN_frag_B, 256)")
    state.add_edge(glb_to_vecin1, None, tasklet, "IN_frag_A", dace.memlet.Memlet(f"frag_A[i + ii:i + ii, j + jj:j+jj + 256]"))
    state.add_edge(glb_to_vecin2, None, tasklet, "IN_frag_B", dace.memlet.Memlet(f"frag_B[i + ii:i + ii, j + jj:j+jj + 256]"))
    state.add_edge(tasklet, "OUT_frag_C", libnode, None, dace.memlet.Memlet(f"frag_C[i + ii:i + ii, j + jj:j+jj + 256]"))


    state.add_edge(libnode, None, tblock_exit, "IN_C", dace.memlet.Memlet(f"ascend_C[i + ii:i + ii, j + jj:j+jj + 256]"))
    state.add_edge(tblock_exit, "OUT_C", dev_exit, "IN_C", dace.memlet.Memlet(f"ascend_C[i + ii:i + ii, j + jj:j+jj + 256]"))
    state.add_edge(dev_exit, "OUT_C", cdc, None, dace.memlet.Memlet(f"ascend_C[0:{256*32},0:{256*32}]"))
    state.add_edge(cdc, None, chc, None, dace.memlet.Memlet(f"C[0:{256*32}, 0:{256*32}]"))

    for n in [dev_entry, tblock_entry]:
        n.add_in_connector("IN_A")
        n.add_out_connector("OUT_A")
        n.add_in_connector("IN_B")
        n.add_out_connector("OUT_B")
    for n in [dev_exit, tblock_exit]:
        n.add_in_connector("IN_C")
        n.add_out_connector("OUT_C")

    sdfg.save("ascend_4.sdfgz")
    return sdfg


s = three()
s.compile()

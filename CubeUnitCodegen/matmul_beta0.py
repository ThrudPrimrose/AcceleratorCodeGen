import dace
from dace.transformation.auto_tile.insert_transfers import InsertTransfers
from dace.transformation.auto_tile.replace_scalar_to_vector_unit import ReplaceScalarToVectorUnit
from dace.transformation.auto_tile.thread_coarsening import ThreadCoarsening
from dace.transformation.auto_tile.explicit_memory_move import ExplicitMemoryMove
from dace.transformation.auto_tile.add_compute_element_map import AddComputeElementBlockMap
from dace.transformation.auto_tile.block_tiling import BlockTiling
from dace.transformation.auto_tile.consecutive_block_tiling import ConsecutiveBlockTiling
import json
import numpy as np
from dace.sdfg import utils as sdutil

N = dace.symbol("N")
M = dace.symbol("M")
K = dace.symbol("K")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    choices=["ascend910", "ascend910B4"],
    default="ascend910B4",
    help="Target device type (ascend910 or ascend910B4). Default is ascend910B4."
)
args = parser.parse_args()

# Set the DaCe config based on input
dace.config.Config.set("compiler", "ascendc", "soc_version", value=args.device)

if args.device == "ascend910":
    NUM_AI_CORES = 32
elif args.device == "ascend910B4":
    NUM_AI_CORES = 20
else:
    raise ValueError("Invalid device type. Choose either 'ascend910' or 'ascend910B4'.")


import networkx as nx

glb = str(dace.dtypes.StorageType.Ascend_Global)
a2 = str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_A2)
b2 = str(dace.dtypes.StorageType.Ascend_L2) + "@" +  str(dace.dtypes.StorageType.Ascend_B2)
a1 = str(dace.dtypes.StorageType.Ascend_L1) + "@" +  str(dace.dtypes.StorageType.Ascend_A1)
b1 = str(dace.dtypes.StorageType.Ascend_L1) + "@" +  str(dace.dtypes.StorageType.Ascend_B1)
co2 = str(dace.dtypes.StorageType.Ascend_L2) + "@" +  str(dace.dtypes.StorageType.Ascend_CO2)
co1 = str(dace.dtypes.StorageType.Ascend_L1) + "@" +  str(dace.dtypes.StorageType.Ascend_CO1)
vecin = str(dace.dtypes.StorageType.Ascend_VECIN)
vecout = str(dace.dtypes.StorageType.Ascend_VECOUT)
nodes = [glb, a2, b2, a1, b1, co2, co1]
finf = float('inf')

# Memory locations, and possible paths for Ascend
graph = {
    glb: { glb: 0, a2: 1, b2: 1, a1 : finf, b1: finf, co2 : finf, co1: 1, vecin: 1, vecout: finf },
    a2: { glb: finf, a2: 0, b2: finf, a1 : 1, b1: finf, co2 : finf, co1: finf, vecin: finf, vecout: finf },
    b2: { glb: finf, a2: finf, b2: 0, a1 : finf, b1: 1, co2 : finf, co1: finf, vecin: finf, vecout: finf },
    a1: { glb: finf, a2: finf, b2: finf, a1 : 0, b1: finf, co2 : 1, co1: finf, vecin: finf, vecout: finf },
    b1: { glb: finf, a2: finf, b2: finf, a1 : finf, b1: 0, co2 : 1, co1: finf, vecin: finf, vecout: finf} ,
    co2: { glb: 1, a2: finf, b2: finf, a1 : finf, b1: finf, co2 : 0, co1: finf, vecin: 1, vecout: finf },
    co1: { glb: finf, a2: finf, b2: finf, a1 : finf, b1: finf, co2 : 1, co1: 0, vecin: finf, vecout: finf },
    vecin: { glb: finf, a2: finf, b2: finf, a1 : finf, b1: finf, co2 : finf, co1: finf, vecin: finf, vecout: 1 },
    vecout: { glb: 1, a2: finf, b2: finf, a1 : finf, b1: finf, co2 : finf, co1: finf, vecin: finf, vecout: finf }
}

entry_location_requirements = {
    "MMU": [a1, b1,],
    "VECTOR": [vecin,],
}

exit_location_requirements = {
    "MMU": [co1,],
    "VECTOR": [vecout,],
}

computational_unit_register_locations =  {
    "MMU": co1,
    "VECTOR": vecin,
}

in_out_types = {
    vecin: vecout,
    a2 + "_AND_" + b2: co1
}

l1 = str(dace.dtypes.StorageType.Ascend_L1)
l2 = str(dace.dtypes.StorageType.Ascend_L2)
unspecialized_locations = [l1, l2]

sstorage_specialization = {
    l2 + "_AND_A": a2,
    l2 + "_AND_B": b2,
    l1 + "_AND_A": a1,
    l1 + "_AND_B": b1,
    l1 + "_AND_acc": co1,
}

serialized_G = json.dumps(graph)
serialized_entry_location_requirements = json.dumps(entry_location_requirements)
serialized_exit_location_requirements = json.dumps(exit_location_requirements)
serialized_computational_unit_register_locations = json.dumps(computational_unit_register_locations)
serialized_input_output_types = json.dumps(in_out_types)


@dace.program
def gemm(A: dace.float32[M, K] @ dace.dtypes.StorageType.Ascend_Global,
         B: dace.float32[K, N] @ dace.dtypes.StorageType.Ascend_Global,
         C: dace.float32[M, N] @ dace.dtypes.StorageType.Ascend_Global):
    for i, j in dace.map[0:M, 0:N] @ dace.dtypes.ScheduleType.Ascend_Device:
        tmp = 0.0
        for k in dace.map[0:K] @ dace.dtypes.ScheduleType.Sequential:
            tmp = tmp + A[i, k] * B[k, j]
        C[i, j] = tmp


sdfg = gemm.to_sdfg()
sdfg.simplify()
for arr_name, arr in sdfg.arrays.items():
    if arr.dtype == dace.float32 or arr.dtype == dace.float64:
        arr.dtype = dace.float16
sdfg.save("gemm_base1.sdfg")

def prepropcess(sdfg: dace.SDFG):
    for s in sdfg.states():
        nodes_to_rm = set()
        edges_to_rm = set()
        for n in s.nodes():
            if isinstance(n, dace.nodes.AccessNode):
                in_edges = s.in_edges(n)
                if len(in_edges) > 0:
                    in_edge = in_edges[0]
                    if isinstance(in_edge.src, dace.nodes.Tasklet):
                        in_in_edge = s.in_edges(in_edge.src)[0]
                        if in_in_edge.data.data is None:
                            nodes_to_rm.add(in_edge.src)
                            edges_to_rm.add(in_edge)
                            in_in_edge = s.in_edges(in_edge.src)[0]
                            edges_to_rm.add(in_in_edge)
                            print(in_in_edge, in_edge, in_edge.src)
                            s.add_edge(in_in_edge.src, in_in_edge.src_conn, n, None, dace.memlet.Memlet(None))

        for e in edges_to_rm:
            s.remove_edge(e)
        for n in nodes_to_rm:
            s.remove_node(n)

prepropcess(sdfg)
sdfg.save("gemm_base2.sdfg")
sdfg.validate()

for s in sdfg.states():
    for n in s.nodes():
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device
        ):
            AddComputeElementBlockMap.apply_to(
                sdfg=sdfg,
                verify=False,
                map_entry=n,
                options={
                    "compute_element_group_dims": [5, 4, 1] if args.device == "ascend910B4" else [8, 4, 1],
                    "map_schedule": dace.dtypes.ScheduleType.Ascend_Device,
                    "schedule_to_add": dace.dtypes.ScheduleType.Ascend_AiCoreGroup,
                },
            )

sdfg.save("gemm_w_tblock_map.sdfg")
sdfg.validate()

for s in sdfg.states():
    dmap = None
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
    for n in sdutil.dfs_topological_sort(s, dmap):
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Ascend_AiCoreGroup
            and dmap is not None
        ):
            print(
            ThreadCoarsening.can_be_applied_to(sdfg,
                thread_group_map_entry=n,
                device_map_entry=dmap,
                options={
                    "tile_sizes": [32, 32, 1],
                },)
            )
            ThreadCoarsening.apply_to(
                sdfg=sdfg,
                verify=False,
                thread_group_map_entry=n,
                device_map_entry=dmap,
                options={
                    "tile_sizes": [32, 32, 1],
                },
            )
            dmap = None


sdfg.save("gemm_thread_coarsened.sdfg")
sdfg.validate()


for s in sdfg.states():
    dmap = None
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
    work_map = None
    for n in sdutil.dfs_topological_sort(s, dmap):
        if (isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Sequential
            and n.map.label != "ThreadCoarsenedMap"):
            work_map = n
            break
    for n in sdutil.dfs_topological_sort(s, dmap):
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.map.schedule == dace.dtypes.ScheduleType.Ascend_AiCoreGroup
            and dmap is not None
            and work_map is not None
        ):
            print(work_map)
            print(
            BlockTiling.can_be_applied_to(sdfg,
                thread_block_map_entry=n,
                work_map_entry=work_map,
                options={
                    "block_tile_sizes": (32,),
                },)
            )
            sdfg.validate()
            BlockTiling.apply_to(
                sdfg=sdfg,
                verify=False,
                thread_block_map_entry=n,
                work_map_entry=work_map,
                options={
                    "block_tile_sizes": (32,),
                },
            )
            sdfg.validate()
            dmap = None

sdfg.save("gemm_block_tiled.sdfg")

for s in sdfg.states():
    dmap = None
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
    for n in sdutil.dfs_topological_sort(s, dmap):
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.label.startswith("OuterWorkMap")
            and dmap is not None
        ):
            print(n)
            print(
            ConsecutiveBlockTiling.can_be_applied_to(sdfg,
                block_tiled_map_entry=n,
                options={
                    "level": 1,
                    "block_tile_factor": (1,),
                },)
            )
            sdfg.validate()
            ConsecutiveBlockTiling.apply_to(
                sdfg=sdfg,
                verify=False,
                block_tiled_map_entry=n,
                options={
                    "level": 1,
                    "block_tile_factor": (1,),
                },
            )
            sdfg.validate()
            dmap = None


sdfg.save("gemm_block_tiled_twice.sdfg")


for s in sdfg.states():
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
            ReplaceScalarToVectorUnit.apply_to(
                sdfg=sdfg,
                verify=False,
                device_map_entry=dmap,
            )

sdfg.save("gemm_tasklets_replaced.sdfg")

for s in sdfg.states():
    dmap = None
    tmap = None
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
        elif isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_AiCoreGroup:
            tmap = n
    print("DT", dmap, tmap)
    for n in sdutil.dfs_topological_sort(s, dmap):
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.label.startswith("OuterWorkMapNo")
            and dmap is not None
            and tmap is not None
        ):
            print("1", dmap, tmap, n)
            ExplicitMemoryMove().apply_to(
                sdfg=sdfg,
                verify=False,
                thread_group_map_entry=tmap,
                map_entry=n,
                device_map_entry=dmap,
                options={
                    "tiles_evenly": True,
                    "dst_memory_location": dace.dtypes.StorageType.Ascend_L1,
                    "src_memory_location": dace.dtypes.StorageType.Ascend_Global,
                    "use_lib_node": False,
                    "pad_contig_dim": False,

                    "location_prefixes": {
                        dace.dtypes.StorageType.Ascend_L2: "L2",
                        dace.dtypes.StorageType.Ascend_L1: "L1",
                        dace.dtypes.StorageType.Ascend_Global: "glb",
                        dace.dtypes.StorageType.Ascend_A2: "A2",
                        dace.dtypes.StorageType.Ascend_A1: "A1",
                        dace.dtypes.StorageType.Ascend_B2: "B2",
                        dace.dtypes.StorageType.Ascend_B1: "B1",
                        dace.dtypes.StorageType.Ascend_CO2: "CO2",
                        dace.dtypes.StorageType.Ascend_CO1: "CO1",
                    },
                    "locations_with_purpose": {
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_A2): dace.dtypes.StorageType.Ascend_A2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_B2): dace.dtypes.StorageType.Ascend_B2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_A1): dace.dtypes.StorageType.Ascend_A1,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_B1): dace.dtypes.StorageType.Ascend_B1,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_CO2): dace.dtypes.StorageType.Ascend_CO2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_CO1): dace.dtypes.StorageType.Ascend_CO1,
                    },
                    "prepend_purpose_to_name": True,
                    "level": 0,
                },
            )
            sdfg.save("gemm_mem_moved_p1.sdfg")

    for n in sdutil.dfs_topological_sort(s, dmap):
        if (
            isinstance(n, dace.nodes.MapEntry)
            and n.label.startswith("OuterWorkMapLevel1No0")
            and dmap is not None
            and tmap is not None
        ):
            print("2", dmap, tmap, n)
            ExplicitMemoryMove().apply_to(
                sdfg=sdfg,
                verify=False,
                thread_group_map_entry=tmap,
                map_entry=n,
                device_map_entry=dmap,
                options={
                    "tiles_evenly": True,
                    "dst_memory_location": dace.dtypes.StorageType.Ascend_L2,
                    "src_memory_location": dace.dtypes.StorageType.Ascend_L1,
                    "use_lib_node": False,
                    "pad_contig_dim": False,
                    "location_prefixes": {
                        dace.dtypes.StorageType.Ascend_L2: "L2",
                        dace.dtypes.StorageType.Ascend_L1: "L1",
                        dace.dtypes.StorageType.Ascend_Global: "glb",
                        dace.dtypes.StorageType.Ascend_A2: "A2",
                        dace.dtypes.StorageType.Ascend_A1: "A1",
                        dace.dtypes.StorageType.Ascend_B2: "B2",
                        dace.dtypes.StorageType.Ascend_B1: "B1",
                        dace.dtypes.StorageType.Ascend_CO2: "CO2",
                        dace.dtypes.StorageType.Ascend_CO1: "CO1",
                    },
                    "locations_with_purpose": {
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_A2): dace.dtypes.StorageType.Ascend_A2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_B2): dace.dtypes.StorageType.Ascend_B2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_A1): dace.dtypes.StorageType.Ascend_A1,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_B1): dace.dtypes.StorageType.Ascend_B1,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_CO2): dace.dtypes.StorageType.Ascend_CO2,
                        str(dace.dtypes.StorageType.Ascend_L2) + "@" + str(dace.dtypes.StorageType.Ascend_CO1): dace.dtypes.StorageType.Ascend_CO1,
                    },
                    "prepend_purpose_to_name": True,
                    "level": 1,
                },

            )
            sdfg.save("gemm_mem_moved_p2.sdfg")


sdfg.save("gemm_mem_moved.sdfg")
sdfg.validate()

InsertTransfers(movement_graph=serialized_G,
                entry_location_requirements=serialized_entry_location_requirements,
                exit_location_requirements=serialized_exit_location_requirements,
                computational_unit_register_locations=serialized_computational_unit_register_locations,
                input_output_types=serialized_input_output_types,
                unspecialized_locations=unspecialized_locations,
                dev_entry_type=dace.dtypes.ScheduleType.Ascend_Device).apply_pass(
    sdfg=sdfg,
    pipeline_results={},
)

sdfg.save("gemm_insert_transfers.sdfg")



# Remainder Loop Transformation Broken Right now
"""
for s in sdfg.states():
    dmap = None
    tmap = None
    for n in s.nodes():
        if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_Device:
            dmap = n
        elif isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.dtypes.ScheduleType.Ascend_AiCoreGroup:
            tmap = n

    if dmap is not None and tmap is not None:
        first_inner_work_map = find_node_by_cond(
            s,
            tmap,
            lambda n: isinstance(n, dace.nodes.MapEntry)
            and n.map.label.startswith("InnerWorkMap"),
        )


        RemainderLoop.apply_to(
            sdfg=sdfg,
            verify=True,
            inner_work_map_entry=first_inner_work_map,
            tblock_type=dace.dtypes.ScheduleType.Ascend_AiCoreGroup,
            options={
                "tblock_type": dace.dtypes.ScheduleType.Ascend_AiCoreGroup,
            }
        )
        dmap = None
        tmap = None

sdfg.save("gemm_remainder_loop.sdfg")
"""

sdfg.generate_code()
csdfg = sdfg.compile()

iM = 5*32
iN = 4*32
iK = 32
A = np.random.rand(iM, iK).astype(np.float16)
B = np.random.rand(iK, iN).astype(np.float16)
C = np.zeros((iM, iN), dtype=np.float16)
C_ref = A @ B
print(C_ref)
csdfg(A=A, B=B, C=C, M=iM, N=iN, K=iK)
print(C)
C_diff = np.abs(C - C_ref)
print(C_diff)

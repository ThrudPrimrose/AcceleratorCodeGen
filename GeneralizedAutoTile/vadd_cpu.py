import dace
from dace.transformation.dataflow import MapInterchange

N = dace.symbol('N')

@dace.program
def vadd1(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.CPU_Multicore:
        C[i] = A[i] + B[i]


@dace.program
def vadd2(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N]):
    for _ in dace.map[0:N:32]  @ dace.ScheduleType.Sequential:
        for i in dace.map[0:32] @ dace.ScheduleType.CPU_Persistent:
            C[i] = A[i] + B[i]


sdfg = vadd1.to_sdfg()
sdfg.save("vadd1.sdfg")

sdfg = vadd2.to_sdfg()
sdfg.save("vadd2.sdfg")
sdfg.compile()

for cfg in sdfg.nodes():
    for n in cfg.nodes():
        if isinstance(n, dace.sdfg.nodes.MapEntry):
            if n.map.schedule == dace.ScheduleType.CPU_Persistent:
                i_entry = n
            if n.map.schedule == dace.ScheduleType.Sequential:
                o_entry = n

MapInterchange.apply_to(sdfg,
                        outer_map_entry=o_entry,
                        inner_map_entry=i_entry)

sdfg.save("vadd3.sdfg")
sdfg.compile()


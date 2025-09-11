import dace
from dace.transformation.interstate import LoopToMap
from dace.transformation.dataflow import MapFusion

sdfg = dace.SDFG.from_file("update_tendancies.sdfg")
sdfg.apply_transformations_repeated(LoopToMap)
sdfg.apply_transformations_repeated(MapFusion)

sdfg.save("opt1.sdfg")



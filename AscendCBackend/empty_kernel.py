import dace
import shutil
import os
import csv
import numpy as np

def three(name):
    sdfg = dace.SDFG(name)
    state = sdfg.add_state("empty_main")

    dev_entry, dev_exit = state.add_map(name="copy_map_outer", ndrange={"i": dace.subsets.Range([(0, 32, 1)])}, schedule=dace.dtypes.ScheduleType.Ascend_Device)
    dev_entry.instrument = dace.InstrumentationType.Timer


    sdfg.save("ascend_3.sdfgz")
    #sdfg.instrument = dace.InstrumentationType.Timer
    return sdfg

def warmup():
    sdfg = three(name=f"empty_kernel_warmup")
    sdfg()
    report = sdfg.get_latest_report()
    kernel_time = report.events[0].duration  # Assuming the first event is the kernel time
    del sdfg
warmup()

# Loop through the vector sizes (from 8192 * 1 to 8192 * 1024 in increments of 32)
# Warm up
j = 0
for i in range(0,1,1):
    sdfg = three(name=f"empty_kernel_s{i}")
    sdfg()
    report = sdfg.get_latest_report()
    kernel_time = report.events[0].duration  # Assuming the first event is the kernel time
    del sdfg
    print(f"Kernel time {kernel_time} ms")

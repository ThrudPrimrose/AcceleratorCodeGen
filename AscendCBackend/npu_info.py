import subprocess
import os

# Define all possible commands, types, and options
base_command = ["npu-smi", "info"]
types = [
    "board", "flash", "memory", "usages", "sensors", "temp", "power", "volt", "mac-addr",
    "common", "health", "product", "ecc", "ip", "sys-time", "i2c_check", "work-mode",
    "ecc-enable", "p2p-enable", "ssh-enable", "license", "customized-info", "device-share",
    "nve-level", "aicpu-config", "pcie-err", "mcu-monitor", "err-count", "boot-area",
    "vnpu-mode", "info-vnpu", "vnpu-svm", "cpu-num-cfg", "first-power-on-date",
    "proc-mem", "phyid-remap", "vnpu-cfg-recover", "key-manage", "template-info",
    "pkcs-enable", "p2p-mem-cfg", "pwm-mode", "pwm-duty-ratio", "boot-select", "topo"
]
options = ["-m", "-l", "-h"]
output_log = "npu_smi_output.log"

# Function to execute a command and collect output
def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command {cmd}: {e.stderr}"

# Collect all outputs
if not os.path.exists(output_log):
    with open(output_log, "w") as log_file:
        for i in range(1): # Extend this by parsing -m
            # Run commands for -t type
            for t in types:
                command = base_command + ["-i", str(i), "-t", t]
                output = run_command(command)
                log_file.write(f"Command: {' '.join(command)}\n{output}\n")

            # Run commands for -m, -l, -h
            for opt in options:
                command = base_command + [opt]
                output = run_command(command)
                log_file.write(f"Command: {' '.join(command)}\n{output}\n")

print(f"Output collected in {output_log}")



def parse_device_info(file_path):
    devices = []
    in_region = False

    with open(file_path, "r") as log_file:
        for line in log_file:
            # Identify the start of the relevant section
            if "Command: npu-smi info -m" in line:
                in_region = True
                continue

            # Process lines within the section
            if in_region:
                # Skip header lines
                if "NPU ID" in line and "Chip ID" in line:
                    continue

                # End of the region: empty line or unrelated content
                if line.strip() == "" or line.startswith("Command:"):
                    in_region = False
                    break

                # Parse device information
                parts = line.split()
                if len(parts) >= 4:  # Ensure the line has enough columns
                    device_id = int(parts[0])  # NPU ID
                    device_name = parts[3] + " " + parts[4]     # Chip Name
                    devices.append((device_id, device_name))

    return devices


def parse_peak_flops(file_path, devices):
    cube_unit_flops = 16 * 16 * 16 * 2
    with open(file_path, "r") as log_file:
        dev_id = devices[0][0]
        aicore_freq = None
        aicore_count = None
        in_region = False
        for line in log_file:
            # Split the output into lines and process each line
            if f"Command: npu-smi info -i {dev_id}" in line and "-t common" in line:
                in_region = True
                continue
            if in_region:
                if "Aicore Freq(MHZ)" in line:
                    aicore_freq = int(line.split(':')[-1].strip())
                    continue
                elif "Aicore Count" in line:
                    aicore_count = int(line.split(':')[-1].strip())
                    continue
                elif "NPU Real-time Power" in line:
                    in_region = False
                    continue

            if not in_region:
                if aicore_count is not None and aicore_freq is not None:
                    break

        dev_name = devices[0]
        print("Device Name:", dev_name[1], ", AiCore Freq (MHz):", aicore_freq, ", #AiCores:", aicore_count)
        print("Theorethical FP16 flops: ", aicore_freq * 1e6 * aicore_count * cube_unit_flops * 1e-12, "TFlop/s")
    return dev_name[1], aicore_freq, aicore_count, aicore_freq * 1e6 * aicore_count * cube_unit_flops * 1e-12


def parse_clock_speeds(file_path, devices):
    ddr_clock_speed = None
    hbm_clock_speed = None
    in_region = False

    with open(file_path, "r") as log_file:
        dev_id = devices[0][0]
        for line in log_file:
            # Identify the start of the relevant section
            if f"Command: npu-smi info -i {dev_id}" in line and "-t memory" in line:
                in_region = True
                continue

            # Process lines within the region
            if in_region:
                # Extract DDR Clock Speed
                if "DDR Clock Speed(MHz)" in line:
                    ddr_clock_speed = int(line.split(':')[-1].strip())
                    continue

                # Extract HBM Clock Speed
                if "HBM Clock Speed(MHz)" in line:
                    hbm_clock_speed = int(line.split(':')[-1].strip())
                    continue

                # End the region when encountering a blank line
                if "Chip ID" in line:
                    in_region = False

    print(f"DDR Clock Speed(MHz): {ddr_clock_speed}, HBM Clock Speed(MHz): {hbm_clock_speed}")
    return ddr_clock_speed, hbm_clock_speed

devices = parse_device_info(output_log)
dev_name, aicore_freq, aicore_count, peak_tflops = parse_peak_flops(output_log, devices)
ddr_clock_speed, hbm_clock_speed = parse_clock_speeds(output_log, devices)

theo = 1228.0
vadd_size = 2048 / 8
print(f"HBM Clock speed: {hbm_clock_speed * 1e6 * 1e-9} Hz")
print(f"Theorethical Peak Bandwidth from Huawei: {theo} GB/s")
print(f"Memory BusWidth = {theo / (hbm_clock_speed * 1e6 * 1e-9)} Bytes?")

# 3 16x16 tiles, half (2 byzes)
# cube_input_size = 16 * 16 * 2 * 2
# print(f"Using Vector Unit: {hbm_clock_speed * 1e6 * 1e-9 * aicore_count * vadd_size} GB/s?")
# print(f"Using Cube Unit: {hbm_clock_speed * 1e6 * 1e-9 * aicore_count * cube_input_size} GB/s?")

import re

command = "lspci -vv | grep -i memory"
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Split the output into lines
lines = result.stdout.splitlines()

pattern = re.compile(r"Region\s+\d+:\s+Memory\s+at\s+[a-f0-9x]+.*\[size=(32G)\]", re.IGNORECASE)

filtered_lines = [line for line in lines if 'size=32G' in line or 'memory controller' in line.lower()]

# Print the filtered lines
for line in filtered_lines:
    print(line)
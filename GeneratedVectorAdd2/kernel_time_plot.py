import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the binary file containing kernel times
file_paths = ['v1.bin']
for file_path in file_paths:
    try:
        kernel_times = np.fromfile(file_path, dtype=np.float32)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        exit(1)

    # Step 2: Plot the distribution of kernel execution times
    plt.figure(figsize=(10, 6))
    plt.hist(kernel_times, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Kernel Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Kernel Execution Times')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{file_path}.png")

import subprocess
import os

executable_path = './build/runner_npu'  # Replace with the actual path to your compiled program

# Number of times to run the executable
total_executions = 10000
runs_per_execution = 1  # Number of runs to perform in each execution of the executable
kernel_times = np.zeros(total_executions, dtype=np.float32)
binary_output_path = 'all_kernel_times.bin'

single_path = 'output_kt.log'

if not os.path.isfile(single_path):
    for i in range(total_executions):
        try:
            result = subprocess.run([executable_path, str(runs_per_execution)],
                                    capture_output=True, text=True, check=True)
            with open('empty_kernel_times.bin', 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)

            if len(data) != 1:
                print(f"Warning: Unexpected data size in binary file at iteration {i + 1}")
            kernel_times[i] = data[0] if len(data) == 1 else 0.0
            print(i, kernel_times[i])
            with open(binary_output_path, 'ab') as binary_file:
                np.array([kernel_times[i]], dtype=np.float32).tofile(binary_file)
            if (i + 1) % 100 == 0:  # Print progress every 100 iterations
                print(f"Progress: {i + 1}/{total_executions} executions completed")
        except subprocess.CalledProcessError as e:
            print(f"Error during execution {i + 1}: {e.stderr}")
            break

import re
from scipy.stats import zscore
import seaborn as sns

pattern = r'^\d+\s+(\d*\.\d+)$'

with open(single_path, 'r') as file:
    input_text = "\n".join(file.readlines())  # Reads all lines as a list of strings

    # Extract floats from matching lines
    float_list = [float(match.group(1)) for match in re.finditer(pattern, input_text, re.MULTILINE)]

    print(float_list)

    # Calculate the z-scores for the data
    data = float_list
    z_scores = zscore(data)

    # Identify outliers (values with |z| > 3)
    outliers = [x for x, z in zip(data, z_scores) if abs(z) > 3]
    filtered_data = [x for x, z in zip(data, z_scores) if abs(z) <= 3]


    print("Data:", data)
    print("Z-scores:", z_scores)
    print("Outliers:", outliers)

    # Step 2: Plot the distribution of kernel execution times
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Kernel Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Kernel Execution Times')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{file_path}.png")

    plt.figure(figsize=(8, 5))

    # Histogram + KDE plot to visualize the distribution
    sns.histplot(filtered_data, kde=True, color='skyblue', bins=100, label='Filtered Data')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of ')
    plt.legend()
    plt.savefig(f"{file_path}2.png")

    # Histogram + KDE plot to visualize the distribution
    sns.histplot(filtered_data, kde=True, color='skyblue', bins=100, label='Filtered Data')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Filtered Data (Outliers Removed)')
    plt.legend()
    plt.savefig(f"{file_path}2_with_outliers.png")

    plt.figure(figsize=(10, 6))
    plt.hist(filtered_data, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Kernel Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Kernel Execution Times')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{file_path}.png")

    plt.figure(figsize=(8, 5))

    # Histogram + KDE plot to visualize the distribution
    sns.histplot(filtered_data, kde=True, color='skyblue', bins=10000, label='Filtered Data')
    plt.savefig(f"{file_path}2_no_bin.png")

peak_mem_bandwidth = 800.0 # GB/s

import pandas as pd
import matplotlib.pyplot as plt
import ast
import io
import numpy as np
import copy


file_path = './kernel_runtimes_no_sync_pipe_all_910b.csv'
df = pd.read_csv(file_path)

# Filter data for frag_size >= 1024
filtered_df = copy.deepcopy(df[df['frag_size'] >= 512])

vector_sizes = filtered_df['vector_size'].unique()
frag_sizes = filtered_df['frag_size'].unique()

# Prepare data for bar plot
median_runtimes = {vector_size: {} for vector_size in vector_sizes}

for vector_size in vector_sizes:
    for frag_size in frag_sizes:
        # Extract the runtimes for the current vector and frag size
        filtered_runtimes = filtered_df[(filtered_df['vector_size'] == vector_size) & (filtered_df['frag_size'] == frag_size)]['runtimes']
        if not filtered_runtimes.empty:
            runtimes = ast.literal_eval(filtered_runtimes.values[0])[2:]  # Skip first 2 entries
            runtimes = [float(x) for x in runtimes]
            median_runtimes[vector_size][frag_size] = np.median(runtimes)

median_runtime_col = []

for _, row in filtered_df.iterrows():
    vector_size = row['vector_size']
    frag_size = row['frag_size']
    median_runtime = median_runtimes.get(vector_size, {}).get(frag_size, np.nan)
    median_runtime_col.append(median_runtime)

# Add the column to the DataFrame
filtered_df['median_runtime'] = median_runtime_col

vector_sizes = sorted(filtered_df['vector_size'].unique())
frag_sizes = sorted(filtered_df['frag_size'].unique())

# Set up bar plotting
bar_width = 0.08
x_indexes = np.arange(len(vector_sizes))

# Create plot
fig, ax1 = plt.subplots(figsize=(10, 8))
ax2 = ax1.twinx()

# Plot bars for each frag_size
for i, frag_size in enumerate(frag_sizes):
    y_vals = []
    bandwidth_vals = []
    for v_size in vector_sizes:
        row = filtered_df[(filtered_df['vector_size'] == v_size) & (filtered_df['frag_size'] == frag_size)]
        if not row.empty:
            y_vals.append(row['median_runtime'].values[0])
            median_runtime_ms = row['median_runtime'].values[0]
            bandwidth = (3 * v_size * 2 / 1e9) / (median_runtime_ms * 1e-3)  # GB/s
            bandwidth_vals.append(bandwidth)
        else:
            y_vals.append(0)  # Or np.nan if you prefer gaps
            bandwidth_vals.append(0)
    x_shifted = x_indexes + i * bar_width
    plt.bar(x_shifted, bandwidth_vals, width=bar_width, label=f'Fragment Size={frag_size}')

# Label formatting
# X-axis setup
ax1.set_xlabel('Vector Size (GB)')
ax1.set_xticks(x_indexes + bar_width * (len(frag_sizes) - 1) / 2)
ax1.set_xticklabels([f"{v*2 / 1e9:.5f}" for v in vector_sizes], rotation=90)

# Left Y-axis (ratio to peak bandwidth)
ax1.set_ylabel('Ratio to Peak Bandwidth')
ax1.set_ylim(0, 1.01)
ax1.set_yticks(np.linspace(0.0, 1.0, 21))
#ax1.set_yscale('log')

# Right Y-axis (absolute achieved bandwidth)
ax2.set_ylabel('Achieved Bandwidth (GB/s)')
ax2.set_ylim(0, peak_mem_bandwidth+1)
ax2.set_yticks(np.linspace(0, peak_mem_bandwidth, 21))
#ax2.set_yscale('log')

# Formatting
ax1.grid(axis="both", which="both", linestyle='--', alpha=0.7)
fig.suptitle('Achieved Bandwidth vs Vector Size for Each Fragment Size')
ax2.legend(title='Frag Size', loc='upper left')

plt.legend(title='Frag Size', loc='upper center',
           bbox_to_anchor=(0.5, -0.18), ncol=3)


plt.tight_layout()
plt.savefig("median_runtime_vs_vector_size.png")


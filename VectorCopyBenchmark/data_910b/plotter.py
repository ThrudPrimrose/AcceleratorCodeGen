peak_mem_bandwidth = 1128.2 # GB/s

import pandas as pd
import matplotlib.pyplot as plt
import ast
import io


file_path = './kernel_runtimes_no_sync_pipe_all.csv'
df = pd.read_csv(file_path)

# Filter data for frag_size = 256
filtered_df = df[df['frag_size'] == 256]

vector_sizes = filtered_df['vector_size'].unique()
for vector_size in vector_sizes:
    # Extract the runtimes for the current vector size
    runtimes = filtered_df[filtered_df['vector_size'] == vector_size]['runtimes'].values[0]
    runtimes = ast.literal_eval(runtimes)[2:]  # Convert string representation of list to an actual list and skip first 2 entries

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(runtimes, bins=10, alpha=0.7, color='steelblue', edgecolor='black')

    plt.xlabel('Runtime (ms)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Runtimes for Vector Size {vector_size} (frag_size=256)')
    #plt.xlim(0, 30)
    #plt.ylim(0, 1.05 * max(plt.hist(runtimes, bins=10, alpha=0.0)[0]))  # Calculate max value from histogram

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"plot_{vector_size}.png")


vector_sizes = df['vector_size'].unique()
frag_sizes = df['frag_size'].unique()

# Prepare data for bar plot
average_runtimes = {vector_size: {} for vector_size in vector_sizes}

for i, decr in enumerate([0, 0.125]):
    for vector_size in vector_sizes:
        for frag_size in frag_sizes:
            # Extract the runtimes for the current vector and frag size
            filtered_runtimes = df[(df['vector_size'] == vector_size) & (df['frag_size'] == frag_size)]['runtimes']
            if not filtered_runtimes.empty:
                runtimes = ast.literal_eval(filtered_runtimes.values[0])[2:]  # Skip first 2 entries
                runtimes = [float(x) for x in runtimes]
                average_runtimes[vector_size][frag_size] = sum(runtimes) / len(runtimes) if runtimes else 0

    # Plot the bar plot
    x = list(average_runtimes.keys())  # Vector sizes
    spacing = 2  # Increase this to add more space between groups
    x_indexes = [i * spacing for i in range(len(x))]  # Increased spacing between groups
    bar_width = 0.15

    plt.figure(figsize=(12, 6))

    #total_bytes = vector_size * 3
    #best_time = (total_bytes * 1e-9) / peak_mem_bandwidth


    for i, frag_size in enumerate(frag_sizes):
        #plt.bar([index + i * bar_width for index in x_indexes], y, width=bar_width, label=f'frag_size={frag_size}')
        best_time = ((x[i] * 3 * 2 * 1e-9) / (peak_mem_bandwidth * 1e-3))
        y = [best_time / average_runtimes[vector_size].get(frag_size, 0) for vector_size in x]
        plt.bar([index + i * bar_width for index in x_indexes], y, width=bar_width, label=f'frag_size={frag_size}')

    best_times = [((x[i] * 3 * 2 * 1e-9) / (peak_mem_bandwidth * 1e-3)) for i in range(len(x))]
    print(best_times)
    #plt.bar([index + len(frag_sizes) * bar_width for index in x_indexes], best_times, width=bar_width, label=f'theo best')
    plt.bar([index + len(frag_sizes) * bar_width for index in x_indexes], 1.0, width=bar_width, label=f'theo best')


    plt.xlabel('Vector Size')
    plt.ylabel('Average Runtime (ms)')
    plt.title('Average Runtimes per Vector Size for Each frag_size')

    plt.xticks([index + (len(frag_sizes) / 2 - 0.5) * bar_width for index in x_indexes], x)
    plt.legend(title='Frag Size')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"bars{i}.png")


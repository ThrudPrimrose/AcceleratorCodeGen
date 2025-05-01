import numpy as np
import glob
import re
import itertools
from collections import defaultdict

def load_matrix_from_file(filepath, N):
    """Loads a binary file and reshapes it to an N x N matrix."""
    data = np.fromfile(filepath, dtype=np.float32)
    return data.reshape(N, N)

def get_matrix_size(filename):
    """Extracts matrix size from the filename."""
    match = re.search(r'C_(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not parse size from filename {filename}")

def find_files():
    """Finds all relevant binary files based on the provided pattern."""
    pattern = re.compile(r"C_\d+_\d+_(cuda|numpy|ascend|aclblas)(_half)?_ref\.bin")
    return [f for f in glob.glob("C_*_*_*_ref.bin") if pattern.search(f)]

def compute_differences(matrices):
    """Calculates absolute differences, average, and variance for all matrix pairs."""
    results = {}
    for (name1, matrix1), (name2, matrix2) in itertools.combinations(matrices.items(), 2):
        diff = np.abs(matrix1 - matrix2)
        avg_diff = np.mean(diff)
        var_diff = np.var(diff)
        results[f"{name1} vs {name2}"] = (avg_diff, var_diff)
    return results

def group_matrices_by_size(matrices):
    """Groups matrices into different groups based on their size."""
    grouped_matrices = defaultdict(dict)
    for filename, matrix in matrices.items():
        N = matrix.shape[0]  # Assuming square matrices
        if N in [16, 64, 4096, 8192]:
            grouped_matrices[N][filename] = matrix
        else:
            print(f"Warning: Matrix size {N} not recognized for file {filename}")
    return grouped_matrices

def main():
    files = find_files()
    matrices = {}

    for filepath in files:
        N = get_matrix_size(filepath)
        matrix = load_matrix_from_file(filepath, N)
        matrices[filepath] = matrix

    # Group matrices by size
    grouped_matrices = group_matrices_by_size(matrices)

    # Compute and print differences for each group
    for group_size, group_matrices in grouped_matrices.items():
        print(f"\nProcessing group of size {group_size}x{group_size} matrices:")
        differences = compute_differences(group_matrices)
        for pair, (avg, var) in differences.items():
            print(f"{pair}: Average Difference = {avg:.6f}, Variance = {var:.6f}")

if __name__ == "__main__":
    main()

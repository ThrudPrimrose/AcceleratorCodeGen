import numpy as np
import glob
import re
import itertools

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
    pattern = re.compile(r"C_\d+_\d+_(cuda|numpy|ascend)(_half)?_ref\.bin")
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

def main():
    files = find_files()
    matrices = {}

    for filepath in files:
        N = get_matrix_size(filepath)
        matrix = load_matrix_from_file(filepath, N)
        matrices[filepath] = matrix

    # Compute and print differences
    differences = compute_differences(matrices)
    print(differences)
    for pair, (avg, var) in differences.items():
        print(f"{pair}: Average Difference = {avg:.6f}, Variance = {var:.6f}")

if __name__ == "__main__":
    main()

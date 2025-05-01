import numpy as np
import glob
import re
import itertools
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

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

def compute_differences(matrices, reference_matrix=None):
    """Calculates absolute and relative differences for all matrix pairs (and against reference if provided)."""
    results = {}
    for filename, matrix in matrices.items():
        if reference_matrix is not None:
            # Compute absolute and relative differences with the reference matrix
            abs_diff = np.abs(matrix - reference_matrix)
            rel_diff = np.abs(abs_diff / (np.abs(matrix) + np.abs(reference_matrix)))
            avg_abs_diff = np.mean(abs_diff)
            avg_rel_diff = np.mean(rel_diff)
            var_abs_diff = np.var(abs_diff)
            var_rel_diff = np.var(rel_diff)
            results[filename] = {
                "avg_abs_diff": avg_abs_diff,
                "avg_rel_diff": avg_rel_diff,
                "var_abs_diff": var_abs_diff,
                "var_rel_diff": var_rel_diff
            }
    return results

def group_matrices_by_size(matrices):
    """Groups matrices into different groups based on their size."""
    grouped_matrices = defaultdict(dict)
    for filename, matrix in matrices.items():
        N = matrix.shape[0]  # Assuming square matrices
        if N in [64, 4096, 8192]:
            grouped_matrices[N][filename] = matrix
        else:
            print(f"Warning: Matrix size {N} not recognized for file {filename}")
    return grouped_matrices

def plot_differences(differences,name):
    """Generates a bar plot for the differences."""
    filenames = list(differences.keys())
    print(differences)
    def extract_impl_type(filename):
        match = re.search(r'C_\d+_\d+_(ascend|aclblas)', filename)
        if match:
            if match.group(1) == "aclblas":
                return "AclBlas"
            else:
                return match.group(1).capitalize()  # Extract either 'ascend' or 'aclblas'
        else:
            return None  # Return None if neither is found
    filename_map = [(f, extract_impl_type(f)) for f in filenames if extract_impl_type(f)]

    avg_abs_diffs = [differences[f]["avg_abs_diff"] for f in filenames]
    avg_rel_diffs = [differences[f]["avg_rel_diff"] for f in filenames]

    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    label_dict = dict(filename_map)
    new_labels = [label_dict.get(label, label) for label in filenames]

    # Plot absolute differences
    ax[0].bar(filenames, avg_abs_diffs, color='blue')
    ax[0].set_title('Average Absolute Differences')
    ax[0].set_ylabel('Average Absolute Difference')
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_xticks(filenames, new_labels)

    # Plot relative differences
    ax[1].bar(filenames, avg_rel_diffs, color='green')
    ax[1].set_title('Average Relative Differences')
    ax[1].set_ylabel('Average Relative Difference')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_xticks(filenames, new_labels)

    plt.tight_layout()
    plt.grid(which="both")
    plt.savefig(name+".png")

def plot_all_differences(all_differences, n):
    plt.clf()
    sorted_dict = list(sorted(list(all_differences.items()), key=lambda item: item[0], reverse=False))
    x_pos = 0
    new_labels = []
    max_a = 0
    for size, differences in sorted_dict:
        """Generates a bar plot for the differences."""
        filenames = list(differences.keys())
        print(differences)
        def extract_impl_type(filename):
            match = re.search(r'C_\d+_\d+_(ascend|aclblas)', filename)
            if match:
                if match.group(1) == "aclblas":
                    return "AclBlas"
                else:
                    return match.group(1).capitalize() + "C"  # Extract either 'ascend' or 'aclblas'
            else:
                return None  # Return None if neither is found
        filename_map = dict([(extract_impl_type(f), f) for f in filenames if extract_impl_type(f)])

        avg_abs_diffs = [differences[f]["avg_abs_diff"] for f in filenames]
        avg_rel_diffs = [differences[f]["avg_rel_diff"] for f in filenames]

        # Create subplots

        label_dict = dict(filename_map)
        new_labels.append("M=N=K=" + str(size))

        # Plot absolute differences

        bar_width = 0.35
        if differences[filename_map["AclBlas"]]["avg_abs_diff"] > max_a:
            max_a = differences[filename_map["AclBlas"]]["avg_abs_diff"]
        bar1 = plt.bar(x_pos - bar_width/2, differences[filename_map["AscendC"]]["avg_abs_diff"], color='#99CCFF', width=bar_width, label='AscendC' if x_pos == 0 else None, align='center', zorder=2)
        bar2 = plt.bar(x_pos + bar_width/2, differences[filename_map["AclBlas"]]["avg_abs_diff"], color='#FF9999', width=bar_width, label='AclBlas' if x_pos == 0 else None, align='center', zorder=2)

        plt.errorbar(x=bar2[0].get_x() + bar2[0].get_width() / 2,  # Position of the bar
                    y=bar2[0].get_height(),  # Top of the bar
                    yerr=differences[filename_map["AclBlas"]]["var_abs_diff"],  # Variance as the error
                    fmt='--',  # Error bar marker type (circle)
                    color='black',  # Error bar color
                    ecolor='black',  # Color for the error bars
                    elinewidth=2,  # Thickness of the error bars
                    capsize=7,  # Width of the caps at the end of error bars
                    zorder=3)  # Ensure it's drawn in front
        plt.errorbar(x=bar1[0].get_x() + bar1[0].get_width() / 2,  # Position of the bar
                    y=bar1[0].get_height(),  # Top of the bar
                    yerr=differences[filename_map["AscendC"]]["var_abs_diff"],  # Variance as the error
                    fmt='--',  # Error bar marker type (circle)
                    color='black',  # Error bar color
                    ecolor='black',  # Color for the error bars
                    elinewidth=2,  # Thickness of the error bars
                    capsize=7,  # Width of the caps at the end of error bars
                    zorder=3)  # Ensure it's drawn in front
        x_pos += 1

    #plt.title('Average Abs Differences')
    plt.ylabel('Average Abs. Difference')
    plt.tick_params(axis='x', rotation=0)
    plt.xticks(range(x_pos), new_labels)
    plt.grid(which="both", linestyle='--', linewidth=0.25)
    plt.legend()
    plt.yscale('log')
    plt.ylim(-1, max_a * 4)
    plt.tight_layout()
    plt.savefig(n+".png")
    plt.savefig(n+".pdf")


def main():
    files = find_files()
    matrices = {}

    # Load matrices
    for n, ss in [("a", ""), ("b", "half_")]:
        for filepath in files:
            N = get_matrix_size(filepath)
            matrix = load_matrix_from_file(filepath, N)
            matrices[filepath] = matrix

        # Group matrices by size
        grouped_matrices = group_matrices_by_size(matrices)

        # Identify the reference matrix (cuda_ref)

        # Compute and print differences for each group
        all_differences = dict()
        for group_size, group_matrices in grouped_matrices.items():
            cuda_ref_matrix = None
            for filename, matrix in group_matrices.items():
                if 'cuda_' + ss + 'ref' in filename and str(group_size) in filename:
                    cuda_ref_matrix = matrix
                    break

            if cuda_ref_matrix is None:
                raise ValueError("Could not find a matrix with 'cuda_ref' in the filename.")

            print(f"\nProcessing group of size {group_size}x{group_size} matrices:")
            filtered_matrices = {filename: matrix for filename, matrix in group_matrices.items() if 'ascend' in filename or 'aclblas' in filename}
            differences = compute_differences(filtered_matrices, reference_matrix=cuda_ref_matrix)
            all_differences[group_size] = differences
            for filename, diff in differences.items():
                print(f"{filename}: Avg Abs Diff = {diff['avg_abs_diff']:.6f}, Avg Rel Diff = {diff['avg_rel_diff']:.6f}")

            # Plot the differences
            # plot_differences(differences, str(group_size))
        plot_all_differences(all_differences, n + ss)

if __name__ == "__main__":
    main()

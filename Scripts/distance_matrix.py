import re
import numpy as np
import pandas as pd

# Input string (only floats and values change)
data = """
Time taken by aclblas: 9.26112e-05 seconds
Between impl. matmul and aclblas matmul
max abs difference is: 0.00779724
max rel difference is: 0.0482109%
Between impl. matmul and numpy matmul
max abs difference is: 7.62939e-06
max rel difference is: 4.52667e-05%
Between impl. matmul and numpy matmul fp 16
max abs difference is: 0.0572948
max rel difference is: 0.367278%
Between aclblas matmul and numpy matmul
max abs difference is: 0.00780106
max rel difference is: 0.0482109%
Between aclblas matmul and numpy fp16 matmul
max abs difference is: 0.0625
max rel difference is: 0.37037%
""".replace("numpy matmul fp 16", "numpy fp16 matmul")

# Regular expression to find pairs and their max absolute differences
pattern = re.compile(r"Between (.*?) and (.*?)\nmax abs difference is: ([\d.e+-]+)")

# Parse the string to extract pairs and max absolute differences
matches = pattern.findall(data)
print(matches)

matmuls = sorted(set([m for match in matches for m in match[:2]]))

# Create a mapping of matmul names to indices
matmul_indices = {name: idx for idx, name in enumerate(matmuls)}

# Initialize an empty distance matrix
num_matmuls = len(matmuls)
distance_matrix = np.full((num_matmuls, num_matmuls), -1.0)

# Populate the distance matrix with the max absolute differences
for mat1, mat2, diff in matches:
    idx1 = matmul_indices[mat1]
    idx2 = matmul_indices[mat2]
    distance_matrix[idx1, idx2] = float(diff)
    distance_matrix[idx2, idx1] = float(diff)  # Symmetric matrix

# Print the distance matrix
print("Distance matrix based on max absolute differences:")
print(distance_matrix)

# Optional: Print matrix with matmul names as labels
df = pd.DataFrame(distance_matrix, index=matmuls, columns=matmuls)
print("\nDistance matrix with labels:")
print(df)


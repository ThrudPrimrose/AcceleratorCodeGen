import numpy as np

# Function to read a binary file containing a 64x64 half-precision matrix
def read_half_precision_matrix(file_path):
    # Read the binary data
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Convert binary data to a NumPy array with the correct dtype and shape
    matrix = np.frombuffer(data, dtype=np.float16).reshape((64, 64)).astype(np.float32)
    
    return matrix

# Paths to the binary files
file1 = 'A.bin'
file2 = 'B.bin'

# Read the matrices
matrix1 = read_half_precision_matrix(file1)
matrix2 = read_half_precision_matrix(file2)

# Perform matrix multiplication
result = np.matmul(matrix1, matrix2)

# Print the result
print(result)

result.astype(np.float32).tofile("C_ref.bin")
np.savetxt("C_ref.txt", result.astype(np.float32))
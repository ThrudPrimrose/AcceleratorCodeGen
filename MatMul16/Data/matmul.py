import numpy as np

def read_half_to_single(file_path):
    # Read the binary data
    with open(file_path, 'rb') as f:
        data = f.read()

    # Convert binary data to a NumPy array with the correct dtype and shape
    matrix = np.frombuffer(data, dtype=np.float16).reshape((64, 64)).astype(np.float32)

    return matrix

def read_half(file_path):
    # Read the binary data
    with open(file_path, 'rb') as f:
        data = f.read()

    # Convert binary data to a NumPy array with the correct dtype and shape
    matrix = np.frombuffer(data, dtype=np.float16).reshape((64, 64))

    return matrix

# Paths to the binary files
file1 = 'A.bin'
file2 = 'B.bin'

# Read the matrices
matrix1 = read_half_to_single(file1)
matrix2 = read_half_to_single(file2)

matrix1_fp16 = read_half(file1)
matrix2_fp16 = read_half(file2)

# Perform matrix multiplication
result = np.matmul(matrix1, matrix2)
result_fp16 = np.matmul(matrix1_fp16, matrix2_fp16)

result.astype(np.float32).tofile("C_ref.bin")
np.savetxt("C_ref.txt", result.astype(np.float32))

result_fp16.tofile("C_half_ref.bin")
np.savetxt("C_half_ref.txt", result.astype(np.float32))
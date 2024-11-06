import numpy as np

N=8192

def read_half_to_single(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    matrix = np.frombuffer(data, dtype=np.float16).reshape((N, N)).astype(np.float32)

    return matrix

def read_half(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    matrix = np.frombuffer(data, dtype=np.float16).reshape((N, N))

    return matrix

file1 = f'A_{N}_{N}.bin'
file2 = f'B_{N}_{N}.bin'

matrix1 = read_half_to_single(file1)
matrix2 = read_half_to_single(file2)

matrix1_fp16 = read_half(file1)
matrix2_fp16 = read_half(file2)

#result = np.matmul(matrix1, matrix2)
result_fp16 = np.matmul(matrix1_fp16, matrix2_fp16)

#result.astype(np.float32).tofile(f"C_{N}_{N}_numpy_ref.bin")

result_fp16.tofile(f"C_{N}_{N}_numpy_half_ref.bin")

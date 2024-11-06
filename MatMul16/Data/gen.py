import numpy as np
import argparse

def generate_and_save_matrix(file_name, M, N):
    matrix = np.random.rand(M, N).astype(np.float16)

    matrix.tofile(file_name)
    print(f"Matrix of size {M}x{N} saved to {file_name}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate MxN matrix with random elements and save to a binary file")
    parser.add_argument("file_name", type=str, help="The name of the binary file to save the matrix")
    parser.add_argument("M", type=int, help="Number of rows in the matrix")
    parser.add_argument("N", type=int, help="Number of columns in the matrix")

    # Parse arguments
    args = parser.parse_args()

    # Generate and save the matrix
    generate_and_save_matrix(args.file_name, args.M, args.N)

if __name__ == "__main__":
    main()
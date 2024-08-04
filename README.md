# HPCAI_lab
Labs from NTHU HPC AI camp
Labs from NTHU HPC AI camp

## CUDA Lab
The CUDA lab contains the code file `MATRIX_MUL_GPU.CU` which demonstrates matrix multiplication using CUDA. The code includes a CUDA kernel `matrixMulKernel` that performs the matrix multiplication on the GPU. The lab also provides a `matrixMultiplyGPU` function that allocates device memory and invokes the CUDA kernel. To compile and run the code, use the command `nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu`.



### Configuration for CUDA in [`MATRIX_MUL_GPU.CU`]

1. **CUDA Kernel**: The CUDA kernel [`matrixMulKernel`] performs the matrix multiplication on the GPU.

2. **Memory Allocation**: The [`matrixMultiplyGPU`] function allocates device memory and invokes the CUDA kernel.

3. **Compilation Command**: To compile the CUDA code, use the command [`nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu`].

### Explanation of [`MATRIX_MUL_GPU.CU`]

The [`MATRIX_MUL_GPU.CU`] file demonstrates matrix multiplication using CUDA. Here is a detailed explanation:

1. **CUDA Kernel [`matrixMulKernel`]**:
    - This kernel function is executed on the GPU.
    - It takes two input matrices and computes their product.
    - Each thread computes one element of the output matrix.

2. **Host Function [`matrixMultiplyGPU`]**:
    - This function is executed on the CPU.
    - It allocates memory on the GPU for the input and output matrices.
    - It copies the input matrices from the host (CPU) to the device (GPU).
    - It launches the [`matrixMulKernel`] on the GPU.
    - It copies the result matrix from the device back to the host.

3. **Compilation and Execution**:
    - The code is compiled using the NVIDIA HPC compiler [`nvc++`]
    - The command [`nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu`] compiles the CUDA code into an executable named [`matrix_mul_gpu`]
    - Running the executable performs the matrix multiplication on the GPU.

### Example Code Snippet


```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplyGPU(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1024;
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Initialize matrices A and B with some values
    // ...

    matrixMultiplyGPU(A, B, C, N);

    // Use the result matrix C
    // ...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
```

This code provides a basic structure for performing matrix multiplication using CUDA. The actual implementation might vary based on specific requirements and optimizations.







## Benchmark Lab
The benchmark lab includes the file `HPL.DAT` which is the input file for the HPLinpack benchmark. It contains the configuration settings for the benchmark, such as the output file name, device output, problem sizes, process mapping, process grids, threshold, panel factorizations, recursive stopping criteria, and broadcast settings.


## MPI Lab

### Introduction to MPI
MPI (Message Passing Interface) is a standardized and portable message-passing system designed to function on parallel computing architectures. It is widely used for parallel programming in high-performance computing environments.

### Installation
To install MPI, you can use the following commands:

#### On Ubuntu:
```sh
sudo apt-get update
sudo apt-get install -y mpich
```

#### On Windows:
1. Download and install Microsoft MPI from the official [Microsoft MPI website](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
2. Follow the installation instructions provided on the website.

### Configuration
After installing MPI, you need to configure it for your lab environment. Here are the steps:

1. **Set Environment Variables**:
   - On Linux, add the following lines to your `.bashrc` or `.bash_profile`:
     ```sh
     export PATH=/usr/local/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
     ```
   - On Windows, add the MPI bin directory to your system PATH.

2. **Verify Installation**:
   - Run the following command to check if MPI is installed correctly:
     ```sh
     mpiexec --version
     ```

### Running MPI Programs
To run an MPI program, use the `mpiexec` command followed by the number of processes and the program name. For example:

```sh
mpiexec -n 4 ./your_mpi_program
```

This command runs `your_mpi_program` using 4 processes.

### Example
Here is a simple example of an MPI program in C:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
```

To compile and run this program:

1. Save the code to a file, e.g., `hello_mpi.c`.
2. Compile the program:
   ```sh
   mpicc -o hello_mpi hello_mpi.c
   ```
3. Run the program:
   ```sh
   mpiexec -n 4 ./hello_mpi
   ```

This will output a message from each process.




## OpenMP Lab
The OpenMP lab includes the code file `MATRIX_SUM.C` which demonstrates matrix summation using OpenMP. The code includes a `mat_mul` function that performs the matrix summation in parallel using OpenMP directives. To compile and run the code, uncomment the necessary lines for including the `omp.h` header and defining the number of CPUs. Then, use the command `gcc -fopenmp matrix_sum.c -o matrix_sum` to compile and `./matrix_sum` to run the code.

## Profiling Lab
The profiling lab does not have a specific file mentioned in the provided context. Please provide more information or the relevant file for further assistance.

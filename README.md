# HPCAI_lab
Labs from NTHU HPC AI camp
Labs from NTHU HPC AI camp

## CUDA Lab
The CUDA lab contains the code file `MATRIX_MUL_GPU.CU` which demonstrates matrix multiplication using CUDA. The code includes a CUDA kernel `matrixMulKernel` that performs the matrix multiplication on the GPU. The lab also provides a `matrixMultiplyGPU` function that allocates device memory and invokes the CUDA kernel. To compile and run the code, use the command `nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu`.

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

```



## OpenMP Lab
The OpenMP lab includes the code file `MATRIX_SUM.C` which demonstrates matrix summation using OpenMP. The code includes a `mat_mul` function that performs the matrix summation in parallel using OpenMP directives. To compile and run the code, uncomment the necessary lines for including the `omp.h` header and defining the number of CPUs. Then, use the command `gcc -fopenmp matrix_sum.c -o matrix_sum` to compile and `./matrix_sum` to run the code.

## Profiling Lab
The profiling lab does not have a specific file mentioned in the provided context. Please provide more information or the relevant file for further assistance.

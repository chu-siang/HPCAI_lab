#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

//TODO: CUDA kernel for matrix multiplication
global void matrixMulKernel(float* A, float* B, float* C, int N) {
    //hint: implement the matrix multiplication
    // blockDim.y is block long
    // blockDim.x is block width
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x; 
    if (row < N && col < N){
        float value = 0;
        for(int k = 0; k < N; k++){
            value += A[row * N + k] * B[k*N+col];
        } 
        C[row * N + col] = value;
    }
}   

void matrixMultiplyGPU(float* h_A, float* h_B, float* h_C, int N) {
    // The matrix size
    size_t size = N * N * sizeof(float);
    
    //TODO: Allocate device(GPU) memory
    //hint: cudaMalloc(??, ??)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    //TODO: Copy matrices from host(CPU) to device(GPU)
    //hint: cudaMemcpy(??, ??, ??, ??)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C, sizeof(float) * size, cudaMemcpyHostToDevice);

    //TODO: Define block and grid dimensions
    //hint: dim3 threadsPerBlock(??, ??)
    //hint: dim3 numBlocks(??, ??)
    // dim3 threadsPerBlock(16);
    dim3 numBlocks(16, 16);
    dim3 threadsPerBlock((N + numBlocks.x-1) / numBlocks.x, (N + numBlocks.y-1) / numBlocks.y);
    //TODO: Launch the matrix multiplication kernel
    //hint: matrixMulKernel<<<??, ??>>>(?, ?, ?, ?)
    matrixMulKernel<<<threadsPerBlock, numBlocks>>>(d_A, d_B, d_C, N);
    //TODO: Copy result from device to host
    //hint: cudaMemcpy(??, ??, ??, ??);
    // cudaMemcpy(h_A, d_A, sizeof(float) * size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_B, d_B, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    //TODO: Free device memory
    //hint: cudaFree()
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 2000; // Example size of the matrix
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Perform matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyGPU(h_A, h_B, h_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Write the last row of the result matrix to a file
    std::ofstream outputFile("gpu_output.txt");
    if (outputFile.is_open()) {
        outputFile << h_C[0] << " ";
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    std::cout << "finish" << std::endl;
    std::cout << "GPU Matrix multiplication took " << duration.count() << " ms\n";

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


// nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu
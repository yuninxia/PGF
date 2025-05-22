#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

float launchAndTimeKernel(dim3 grid, dim3 block, float *A, float *B, float *C, int N)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    matrixMulKernel<<<grid, block>>>(A, B, C, N);
    cudaEventRecord(t1);

    // DON'T call cudaDeviceSynchronize(); events handle that.
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms;                       // milliseconds of kernel *only*
}

// Helper function to print a matrix (for verification)
void printMatrix(float *matrix, int N, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%7.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Helper function to initialize a matrix with random values
void initializeMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0f; // Random values between 0 and 10
    }
}

int main() {
    int N = 2048; // Matrix dimension (N x N)
    int M_SIZE = N * N * sizeof(float); // Size of matrix in bytes

    // Allocate host memory
    float *h_a = (float *)malloc(M_SIZE);
    float *h_b = (float *)malloc(M_SIZE);
    float *h_c = (float *)malloc(M_SIZE); // Host result
    float *h_c_gpu = (float *)malloc(M_SIZE); // Host copy of GPU result

    if (h_a == NULL || h_b == NULL || h_c == NULL || h_c_gpu == NULL) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize host matrices
    srand(0); // Seed for reproducibility
    initializeMatrix(h_a, N);
    initializeMatrix(h_b, N);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&d_a, M_SIZE);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_a!\n"); return 1; }
    cudaStatus = cudaMalloc((void **)&d_b, M_SIZE);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_b!\n"); return 1; }
    cudaStatus = cudaMalloc((void **)&d_c, M_SIZE);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_c!\n"); return 1; }

    // Copy input matrices from host to device
    cudaStatus = cudaMemcpy(d_a, h_a, M_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D for d_a failed!\n"); return 1; }
    cudaStatus = cudaMemcpy(d_b, h_b, M_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D for d_b failed!\n"); return 1; }

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Matrix multiplication %dx%d\n", N, N);
    printf("Grid dimensions: (%d,%d), Block dimensions: (%d,%d)\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Warm-up runs
    const int warmup_runs = 100;
    printf("\nPerforming %d warm-up runs...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; ++i) {
        launchAndTimeKernel(numBlocks, threadsPerBlock, d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    const int benchmark_runs = 100;
    float total_time = 0.0f;
    float min_time = 1e10;
    float max_time = 0.0f;
    
    printf("\nPerforming %d benchmark runs...\n", benchmark_runs);
    for (int i = 0; i < benchmark_runs; ++i) {
        float ms = launchAndTimeKernel(numBlocks, threadsPerBlock, d_a, d_b, d_c, N);
        total_time += ms;
        min_time = min(min_time, ms);
        max_time = max(max_time, ms);
        // if ((i + 1) % 10 == 0) {
        //     printf("Completed %d runs...\n", i + 1);
        // }
    }

    // Calculate statistics
    float avg_time = total_time / benchmark_runs;
    printf("\nBenchmark Results:\n");
    printf("Average time: %.3f ms\n", avg_time);
    printf("Min time:     %.3f ms\n", min_time);
    printf("Max time:     %.3f ms\n", max_time);
    printf("Total runs:   %d\n", benchmark_runs);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // Copy result matrix from device to host
    cudaStatus = cudaMemcpy(h_c_gpu, d_c, M_SIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H for d_c failed!\n"); return 1; }

    printf("\nGPU computation complete.\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_gpu);

    return 0;
} 
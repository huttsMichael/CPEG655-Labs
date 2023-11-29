#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA kernel for matrix multiplication using tiling algorithm
__global__ void matrixMulTiled(float *C, float *A, float *B, int N, int tile_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Identify the starting point of the current tile in the input matrices
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // Dynamic shared memory allocation for storing a tile of matrix A and B
    extern __shared__ float sharedMemory[];
    float *As = (float*)&sharedMemory[0];
    float *Bs = (float*)&sharedMemory[tile_size * tile_size];

    float Cvalue = 0.0f;

    // Loop over tiles of input matrices
    for (int t = 0; t < N / tile_size; ++t) {
        // Load the tiles of matrices A and B into shared memory
        As[ty * tile_size + tx] = A[row * N + t * tile_size + tx];
        Bs[ty * tile_size + tx] = B[(t * tile_size + ty) * N + col];

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the tiles and accumulate the result
        for (int k = 0; k < tile_size; ++k)
            Cvalue += As[ty * tile_size + k] * Bs[k * tile_size + tx];

        // Synchronize before loading the next tiles
        __syncthreads();
    }

    // Write the result to the output matrix
    C[row * N + col] = Cvalue;
}


// Host code for matrix multiplication
void mm(float *C, float *A, float *B, int N) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// Function to calculate Root Mean Square Error (RMSE) between two matrices
float calculateRMSE(float *A, float *B, int size) {
    float sumSquaredDiff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = A[i] - B[i];
        sumSquaredDiff += diff * diff;
    }
    float meanSquaredDiff = sumSquaredDiff / size;
    return sqrtf(meanSquaredDiff);
}

int main(int argc, char **argv) {
    struct timeval begin, end;
    int NB_values[] = {8, 16, 32};
    int NT_values[] = {8, 16, 32};
    
    for (int nbIndex = 0; nbIndex < 3; nbIndex++) {
        for (int ntIndex = 0; ntIndex < 3; ntIndex++) {
            int NB = NB_values[nbIndex];
            int NT = NT_values[ntIndex];

            // Calculate NK based on the relationship N^2 = NB^2 * NT^2 * NK^2
            int NK = sqrt(2048 * 2048 / (NB * NB * NT * NT));

            int N = 2048;
            int size = N * N;

            // Allocate host memory
            float *h_A = (float *)malloc(size * sizeof(float));
            float *h_B = (float *)malloc(size * sizeof(float));
            float *h_C_gpu = (float *)malloc(size * sizeof(float));

            // Allocate device memory
            float *d_A, *d_B, *d_C;
            cudaMalloc((void **)&d_A, size * sizeof(float));
            cudaMalloc((void **)&d_B, size * sizeof(float));
            cudaMalloc((void **)&d_C, size * sizeof(float));

            // Copy host matrices to device
            cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

            // Set up the execution configuration
            dim3 threadsPerBlock(NT, NT);
            dim3 numBlocks(NK, NK);

            // Measure the computation time for GPU version
            gettimeofday(&begin, NULL);

            // Launch the CUDA kernel
            matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N, NB);

            // Wait for the kernel to finish
            cudaDeviceSynchronize();

            gettimeofday(&end, NULL);

            fprintf(stdout, "GPU time for N=%d, NB=%d, NT=%d: %lf\n", N, NB, NT, (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);

            // Copy the result back to the host
            cudaMemcpy(h_C_gpu, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

            // Free device and host memory
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            free(h_A);
            free(h_B);
            free(h_C_gpu);
        }
    }

    return 0;
}
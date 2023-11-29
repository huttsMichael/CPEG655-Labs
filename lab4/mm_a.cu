#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *C, float *A, float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize a variable to store the sum for the current element of matrix C
    float sum = 0.0f;

    // Perform the actual matrix multiplication for the current element (i, j)
    for (int k = 0; k < N; ++k) {
        // Multiply corresponding elements from matrices A and B and accumulate the result
        sum += A[i * N + k] * B[k * N + j];
    }

    // Store the final result in the corresponding element of matrix C
    C[i * N + j] = sum;
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
    // Initialize variable to store the sum of squared differences
    float sumSquaredDiff = 0.0f;

    // Iterate through all elements of the matrices
    for (int i = 0; i < size; i++) {
        // Calculate the difference between corresponding elements of matrices A and B
        float diff = A[i] - B[i];

        // Accumulate the squared difference
        sumSquaredDiff += diff * diff;
    }

    // Calculate the mean squared difference
    float meanSquaredDiff = sumSquaredDiff / size;

    // Calculate the square root of the mean squared difference to get RMSE
    float rmse = sqrtf(meanSquaredDiff);

    // Return the calculated RMSE
    return rmse;
}


int main(int argc, char **argv) {
    struct timeval begin, end;
    int sizes[] = {16, 32};

    // Run for both sizes
    for (int sizeIndex = 0; sizeIndex < 2; sizeIndex++) {
        int N = sizes[sizeIndex];
        int size = N * N;

        // Allocate host memory
        float *h_A = (float *)malloc(size * sizeof(float));
        float *h_B = (float *)malloc(size * sizeof(float));
        float *h_C_cpu = (float *)malloc(size * sizeof(float));
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
        dim3 threadsPerBlock(32, 32); // max threads per block = 1024, 32*32=1024
        dim3 numBlocks(1, 1); // Use only one thread block

        // Measure the computation time for GPU version
        gettimeofday(&begin, NULL);

        // Launch the CUDA kernel
        matrixMul<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N);


        // Measure the computation time for GPU version
        gettimeofday(&begin, NULL);

        // Launch the CUDA kernel
        matrixMul<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N);

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        fprintf(stdout, "GPU time for N=%d: %lf\n", N, (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);

        // Copy the result back to the host
        cudaMemcpy(h_C_gpu, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Measure the computation time for CPU version
        gettimeofday(&begin, NULL);

        // Call the CPU matrix multiplication function
        mm(h_C_cpu, h_A, h_B, N);

        gettimeofday(&end, NULL);

        fprintf(stdout, "CPU time for N=%d: %lf\n", N, (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);

        // Verify the correctness by calculating RMSE
        float rmse = calculateRMSE(h_C_cpu, h_C_gpu, size);
        fprintf(stdout, "RMSE for N=%d: %e\n", N, rmse);

        // Free device and host memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }

    return 0;
}

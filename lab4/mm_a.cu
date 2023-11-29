#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *C, float *A, float *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // initialize a variable to store the sum for the current element of matrix C
    float sum = 0.0f;

    // perform the actual matrix multiplication for the current element (i, j)
    for (int k = 0; k < N; ++k) {
        // multiply corresponding elements from matrices A and B and accumulate the result
        sum += A[i * N + k] * B[k * N + j];
    }

    // store the final result in the corresponding element of matrix C
    C[i * N + j] = sum;
}

// host code for matrix multiplication
void mm(float *C, float *A, float *B, int N) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// function to calculate root mean square error (RMSE) between two matrices
float calculateRMSE(float *A, float *B, int size) {
    // initialize variable to store the sum of squared differences
    float sumSquaredDiff = 0.0f;

    // iterate through all elements of the matrices
    for (int i = 0; i < size; i++) {
        // calculate the difference between corresponding elements of matrices A and B
        float diff = A[i] - B[i];

        // accumulate the squared difference
        sumSquaredDiff += diff * diff;
    }

    // calculate the mean squared difference
    float meanSquaredDiff = sumSquaredDiff / size;

    // calculate the square root of the mean squared difference to get RMSE
    float rmse = sqrtf(meanSquaredDiff);

    // return the calculated RMSE
    return rmse;
}

int main(int argc, char **argv) {
    struct timeval begin, end;
    int sizes[] = {16, 32};
    int num_runs = 5000; 

    for (int sizeIndex = 0; sizeIndex < 2; sizeIndex++) {
        int N = sizes[sizeIndex];
        int size = N * N;

        double total_gpu_time = 0.0;
        double total_cpu_time = 0.0;

        for (int run = 0; run < num_runs; run++) {
            // allocate host memory
            float *h_A = (float *)malloc(size * sizeof(float));
            float *h_B = (float *)malloc(size * sizeof(float));
            float *h_C_cpu = (float *)malloc(size * sizeof(float));
            float *h_C_gpu = (float *)malloc(size * sizeof(float));

            // allocate device memory
            float *d_A, *d_B, *d_C;
            cudaMalloc((void **)&d_A, size * sizeof(float));
            cudaMalloc((void **)&d_B, size * sizeof(float));
            cudaMalloc((void **)&d_C, size * sizeof(float));

            // copy host matrices to device
            cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

            // set up the execution configuration
            dim3 threadsPerBlock(32, 32);
            dim3 numBlocks(1, 1);

            // measure the computation time for GPU version
            gettimeofday(&begin, NULL);

            // launch the CUDA kernel
            matrixMul<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N);

            // wait for the kernel to finish
            cudaDeviceSynchronize();

            gettimeofday(&end, NULL);

            // calculate time down to microsecond
            double elapsed_gpu_time = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000;
            total_gpu_time += elapsed_gpu_time;

            // fprintf(stdout, "run %d - GPU time for N=%d: %lf\n", run + 1, N, elapsed_gpu_time);

            // copy the result back to the host
            cudaMemcpy(h_C_gpu, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

            // measure the computation time for CPU version
            gettimeofday(&begin, NULL);

            // call the CPU matrix multiplication function
            mm(h_C_cpu, h_A, h_B, N);

            gettimeofday(&end, NULL);

            // calculate time down to microsecond
            double elapsed_cpu_time = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000;
            total_cpu_time += elapsed_cpu_time;

            // fprintf(stdout, "Run %d - CPU time for N=%d: %lf\n", run + 1, N, elapsed_cpu_time);

            // verify the correctness by calculating RMSE (commented out for benchmarking)
            // float rmse = calculateRMSE(h_C_cpu, h_C_gpu, size);
            // fprintf(stdout, "RMSE for N=%d: %e\n", N, rmse);

            // free device and host memory
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            free(h_A);
            free(h_B);
            free(h_C_cpu);
            free(h_C_gpu);
        }

        double avg_gpu_time = total_gpu_time / num_runs;
        double avg_cpu_time = total_cpu_time / num_runs;

        fprintf(stdout, "Average GPU time for N=%d: %lf\n", N, avg_gpu_time);
        fprintf(stdout, "Average CPU time for N=%d: %lf\n", N, avg_cpu_time);
    }

    return 0;
}

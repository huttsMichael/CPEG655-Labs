#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA kernel for matrix multiplication using tiling algorithm
__global__ void matrixMulTiled(float *C, float *A, float *B, int N, int tile_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // identify the starting point of the current tile in the input matrices
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // dynamic shared memory allocation for storing a tile of matrix A and B
    extern __shared__ float sharedMemory[];
    float *As = (float*)&sharedMemory[0];
    float *Bs = (float*)&sharedMemory[tile_size * tile_size];

    float Cvalue = 0.0f;

    // loop over tiles of input matrices
    for (int t = 0; t < N / tile_size; ++t) {
        // load the tiles of matrices A and B into shared memory
        As[ty * tile_size + tx] = A[row * N + t * tile_size + tx];
        Bs[ty * tile_size + tx] = B[(t * tile_size + ty) * N + col];

        // synchronize to make sure the tiles are loaded
        __syncthreads();

        // multiply the tiles and accumulate the result
        for (int k = 0; k < tile_size; ++k)
            Cvalue += As[ty * tile_size + k] * Bs[k * tile_size + tx];

        // synchronize before loading the next tiles
        __syncthreads();
    }

    // write the result to the output matrix
    C[row * N + col] = Cvalue;
}


// host code for matrix multiplication
void mm(float *C, float *A, float *B, int N) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

// function to calculate root mean square error 
float calculateRMSE(float *A, float *B, int size) {
    float sumSquaredDiff = 0.0f;

    // iterate through all elements of the matrices
    for (int i = 0; i < size; i++) {
        // calculate the difference between corresponding elements of matrices A and B
        float diff = A[i] - B[i];

        sumSquaredDiff += diff * diff;
    }

    // calculate the mean squared difference
    float meanSquaredDiff = sumSquaredDiff / size;

    // calculate the square root of the mean squared difference to get RMSE
    float rmse = sqrtf(meanSquaredDiff);

    return rmse;
}

int main(int argc, char **argv) {
    struct timeval begin, end;
    int tile_sizes[] = {8, 16};

    // run for both sizes
    for (int sizeIndex = 0; sizeIndex < 2; sizeIndex++) {
        int tile_size = tile_sizes[sizeIndex];

        int N = 2048;
        int size = N * N;

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
        dim3 threadsPerBlock(tile_size, tile_size);
        dim3 numBlocks(N / tile_size, N / tile_size);

        // measure the computation time for GPU version
        gettimeofday(&begin, NULL);

        // launch the CUDA kernel
        matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N, tile_size);

        // wait for the kernel to finish
        cudaDeviceSynchronize();

        gettimeofday(&end, NULL);

        fprintf(stdout, "GPU time for N=%d, tile size=%d: %lf\n", N, tile_size, (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);

        // copy the result back to the host
        cudaMemcpy(h_C_gpu, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

        // measure the computation time for CPU version
        gettimeofday(&begin, NULL);

        // call the CPU matrix multiplication function
        mm(h_C_cpu, h_A, h_B, N);

        gettimeofday(&end, NULL);

        fprintf(stdout, "CPU time for N=%d: %lf\n", N, (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1.0 / 1000000);

        // verify the correctness by calculating RMSE
        float rmse = calculateRMSE(h_C_cpu, h_C_gpu, size);
        fprintf(stdout, "RMSE for N=%d, tile size=%d: %e\n", N, tile_size, rmse);

        // free device and host memory
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

#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include <cmath>

// function to initialize a matrix with random values using C rand()
void initializeRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        // generate a random float value between 1.0 and 10.0
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 9.0 + 1.0;
    }
}

// function to perform SSE-based matrix multiplication
void matrixMultiplySSE(float* A, float* B, float* C, int size) {
    // iterate over each row of matrix A
    for (int i = 0; i < size; i++) {
        // iterate over each column of matrix B 
        for (int j = 0; j < size; j += size) {
            __m128 rowA, vecB;
            __m128 result = _mm_setzero_ps();
            for (int k = 0; k < 4; k++) {
                // load a single element from the current row of A and fill a vector
                rowA = _mm_set1_ps(A[i * size + j + k]);
                // load a vector from the current column of B
                vecB = _mm_loadu_ps(B + k * size + j);

                // multiply the rowA vector with the loaded vecB vector element-wise and add to the result
                result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
            }

            // store the result vector to the current position in matrix C
            _mm_storeu_ps(C + i * size + j, result);
        }
    }
}

// function to perform matrix multiplication without vectorization
void matrixMultiply(const float* A, const float* B, float* C, int size) {
    // iterate over each row of matrix A
    for (int i = 0; i < size; i++) {
        // iterate over each column of matrix B
        for (int j = 0; j < size; j++) {
            // initialize the sum for the current position in matrix C
            float sum = 0.0;

            // iterate over each element in the row of A and column of B
            for (int k = 0; k < size; k++) {
                // multiply the corresponding elements and accumulate the result in the sum
                sum += A[i * size + k] * B[k * size + j];
            }

            // store the accumulated sum in the current position of matrix C
            C[i * size + j] = sum;
        }
    }
}

// function to print a matrix
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// function to calculate root mean square error
float calculateRMSE(float* A, float* B, int size) {
    float sumSquaredDiff = 0.0f;

    // iterate through all elements of the matrices
    for (int i = 0; i < size * size; i++) {
        // calculate the difference between corresponding elements of matrices A and B
        float diff = A[i] - B[i];

        sumSquaredDiff += diff * diff;
    }

    // calculate the mean squared difference
    float meanSquaredDiff = sumSquaredDiff / (size * size);

    // calculate the square root of the mean squared difference to get RMSE
    float rmse = std::sqrt(meanSquaredDiff);

    return rmse;
}

int main() {
    const int size = 4;

    float A[size * size];
    float B[size * size];

    // initialize matrices A and B with random values
    initializeRandomMatrix(A, size);
    initializeRandomMatrix(B, size);

    float C_sse[size * size] = {0.0};
    float C_non_sse[size * size] = {0.0};

    matrixMultiplySSE(A, B, C_sse, size);
    matrixMultiply(A, B, C_non_sse, size);

    // print the results
    std::cout << "Matrix A:\n";
    printMatrix(A, size, size);

    std::cout << "Matrix B:\n";
    printMatrix(B, size, size);

    std::cout << "SSE-based Matrix Multiplication Result:\n";
    printMatrix(C_sse, size, size);

    std::cout << "Non-SSE Matrix Multiplication Result:\n";
    printMatrix(C_non_sse, size, size);

    // calculate and print RMSE
    float rmse = calculateRMSE(C_sse, C_non_sse, size);
    std::cout << "Root Mean Square Error (RMSE): " << rmse << std::endl;

    return 0;
}

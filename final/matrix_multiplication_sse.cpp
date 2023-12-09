#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include <cmath>

// Function to initialize a matrix with random values using C rand()
void initializeRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        // Generate a random float value between 1.0 and 10.0
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 9.0 + 1.0;
    }
}

// Function to perform SSE-based matrix multiplication
void matrixMultiplySSE(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 4) {
            __m128 rowA = _mm_set1_ps(A[i * size + j]);
            __m128 vecB = _mm_loadu_ps(B + j);

            __m128 result = _mm_mul_ps(rowA, vecB);

            for (int k = 1; k < 4; k++) {
                rowA = _mm_set1_ps(A[i * size + j + k]);
                vecB = _mm_loadu_ps(B + k * size + j);

                result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
            }

            _mm_storeu_ps(C + i * size + j, result);
        }
    }
}

// Function to perform matrix multiplication without vectorization
void matrixMultiply(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

// Function to print a matrix
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to calculate root mean square error
float calculateRMSE(float* A, float* B, int size) {
    float sumSquaredDiff = 0.0f;

    // Iterate through all elements of the matrices
    for (int i = 0; i < size * size; i++) {
        // Calculate the difference between corresponding elements of matrices A and B
        float diff = A[i] - B[i];

        sumSquaredDiff += diff * diff;
    }

    // Calculate the mean squared difference
    float meanSquaredDiff = sumSquaredDiff / (size * size);

    // Calculate the square root of the mean squared difference to get RMSE
    float rmse = std::sqrt(meanSquaredDiff);

    return rmse;
}

int main() {
    const int size = 4;

    float A[size * size];
    float B[size * size];

    // Initialize matrices A and B with random values
    initializeRandomMatrix(A, size);
    initializeRandomMatrix(B, size);

    float C_sse[size * size] = {0.0};
    float C_non_sse[size * size] = {0.0};

    matrixMultiplySSE(A, B, C_sse, size);
    matrixMultiply(A, B, C_non_sse, size);

    // Print the results
    std::cout << "Matrix A:\n";
    printMatrix(A, size, size);

    std::cout << "Matrix B:\n";
    printMatrix(B, size, size);

    std::cout << "SSE-based Matrix Multiplication Result:\n";
    printMatrix(C_sse, size, size);

    std::cout << "Non-SSE Matrix Multiplication Result:\n";
    printMatrix(C_non_sse, size, size);

    // Calculate and print RMSE
    float rmse = calculateRMSE(C_sse, C_non_sse, size);
    std::cout << "Root Mean Square Error (RMSE): " << rmse << std::endl;

    return 0;
}

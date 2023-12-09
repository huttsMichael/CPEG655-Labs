#include <iostream>
#include <immintrin.h>

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

int main() {
    const int size = 4; 

    float A[size * size] = {1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0, 12.0,
                            13.0, 14.0, 15.0, 16.0};

    float B[size * size] = {1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0};

    float C_sse[size * size] = {0.0};
    float C_non_sse[size * size] = {0.0};

    matrixMultiplySSE(A, B, C_sse, size);
    matrixMultiply(A, B, C_non_sse, size);

    // Print the results
    std::cout << "SSE-based Matrix Multiplication Result:\n";
    printMatrix(C_sse, size, size);

    std::cout << "Non-SSE Matrix Multiplication Result:\n";
    printMatrix(C_non_sse, size, size);

    return 0;
}

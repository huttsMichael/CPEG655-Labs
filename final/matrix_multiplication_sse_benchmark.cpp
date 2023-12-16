#include <iostream>
#include <immintrin.h>
// #include <chrono> // chrono not working in WSL?
#include <ctime>
#include <cstdlib>

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
        __m128 rowA, vecB;
        __m128 result = _mm_setzero_ps();
        for (int k = 0; k < size; k++) {
            // load a single element from the current row of A and fill a vector
            rowA = _mm_set1_ps(A[i * size + k]);
            // load a vector from the current column of B
            vecB = _mm_loadu_ps(B + k * size);

            // multiply the rowA vector with the loaded vecB vector element-wise and add to the result
            result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
        }

        // store the result vector to the current position in matrix C
        _mm_storeu_ps(C + i * size, result);
    }
}

// function to perform SSE-based matrix multiplication (unrolled)
void matrixMultiplySSEUnrolled(float* A, float* B, float* C, int size) {
    // iterate over each row of matrix A
    for (int i = 0; i < size; i++) {
        // load column 0 from the current row of A and fill a vector
        __m128 rowA = _mm_set1_ps(A[i * size]);

        // load a vector from the current column of B
        __m128 vecB = _mm_loadu_ps(B);

        // multiply the rowA vector with the loaded vecB vector element-wise
        __m128 result = _mm_mul_ps(rowA, vecB);

        // repeat for column 1
        rowA = _mm_set1_ps(A[i * size + 1]);
        vecB = _mm_loadu_ps(B + (1 * size));
        result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
        // repeat for column 2
        rowA = _mm_set1_ps(A[i * size + 2]);
        vecB = _mm_loadu_ps(B + (2 * size));
        result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
        // repeat for column 3
        rowA = _mm_set1_ps(A[i * size + 3]);
        vecB = _mm_loadu_ps(B + (3 * size));
        result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));

        // store the result vector to the current position in matrix C
        _mm_storeu_ps(C + i * size, result);
    }
}

// function to perform SSE-based matrix multiplication (extremely unrolled)
void matrixMultiplySSEUnrolledExtreme(float* A, float* B, float* C, int size) {
    // load column 0 from the current row of A and fill a vector
    __m128 rowA = _mm_set1_ps(A[0]);

    // load a vector from the current column of B
    __m128 vecB = _mm_loadu_ps(B);

    // multiply the rowA vector with the loaded vecB vector element-wise
    __m128 result = _mm_mul_ps(rowA, vecB);

    // repeat for column 1
    rowA = _mm_set1_ps(A[0 * size + 1]);
    vecB = _mm_loadu_ps(B + (1 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 2
    rowA = _mm_set1_ps(A[0 * size + 2]);
    vecB = _mm_loadu_ps(B + (2 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 3
    rowA = _mm_set1_ps(A[0 * size + 3]);
    vecB = _mm_loadu_ps(B + (3 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));

    // store the result vector to the current position in matrix C
    _mm_storeu_ps(C + 0 * size, result);

    result = _mm_setzero_ps();

    rowA = _mm_set1_ps(A[1 * size + 0]);
    vecB = _mm_loadu_ps(B + (0 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 1
    rowA = _mm_set1_ps(A[1 * size + 1]);
    vecB = _mm_loadu_ps(B + (1 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 2
    rowA = _mm_set1_ps(A[1 * size + 2]);
    vecB = _mm_loadu_ps(B + (2 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 3
    rowA = _mm_set1_ps(A[1 * size + 3]);
    vecB = _mm_loadu_ps(B + (3 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));

    // store the result vector to the current position in matrix C
    _mm_storeu_ps(C + 1 * size, result);

    result = _mm_setzero_ps();

    rowA = _mm_set1_ps(A[2 * size + 0]);
    vecB = _mm_loadu_ps(B + (0 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 1
    rowA = _mm_set1_ps(A[2 * size + 1]);
    vecB = _mm_loadu_ps(B + (1 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 2
    rowA = _mm_set1_ps(A[2 * size + 2]);
    vecB = _mm_loadu_ps(B + (2 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 3
    rowA = _mm_set1_ps(A[2 * size + 3]);
    vecB = _mm_loadu_ps(B + (3 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));

    // store the result vector to the current position in matrix C
    _mm_storeu_ps(C + 2 * size, result);

    result = _mm_setzero_ps();

    rowA = _mm_set1_ps(A[3 * size + 0]);
    vecB = _mm_loadu_ps(B + (0 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 1
    rowA = _mm_set1_ps(A[3 * size + 1]);
    vecB = _mm_loadu_ps(B + (1 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 2
    rowA = _mm_set1_ps(A[3 * size + 2]);
    vecB = _mm_loadu_ps(B + (2 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));
    // repeat for column 3
    rowA = _mm_set1_ps(A[3 * size + 3]);
    vecB = _mm_loadu_ps(B + (3 * size));
    result = _mm_add_ps(result, _mm_mul_ps(rowA, vecB));

    // store the result vector to the current position in matrix C
    _mm_storeu_ps(C + 3 * size, result);
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


int main() {
    const int size = 4;
    const int num_repeats = 10000000;

    float A[size * size];
    float B[size * size];

    srand((unsigned)time(0)); 
    // initialize matrices A and B with random values
    initializeRandomMatrix(A, size);
    initializeRandomMatrix(B, size);

    float C_sse[size * size] = {0.0};
    float C_sse_unrolled[size * size] = {0.0};
    float C_sse_unrolled_extreme[size * size] = {0.0};
    float C_non_sse[size * size] = {0.0};

    // measure SSE-based matrix multiplication time
    clock_t start_sse = clock();
    for (int i = 0; i < num_repeats; ++i) {
        matrixMultiplySSE(A, B, C_sse, size);
    }
    clock_t stop_sse = clock();
    double duration_sse = static_cast<double>(stop_sse - start_sse) / CLOCKS_PER_SEC;
    double avg_duration_sse = duration_sse / num_repeats * 1e6; // convert to microseconds

    // measure SSE-based matrix multiplication with unrolled innermost loop time
    clock_t start_sse_unrolled = clock();
    for (int i = 0; i < num_repeats; ++i) {
        matrixMultiplySSEUnrolled(A, B, C_sse_unrolled, size);
    }
    clock_t stop_sse_unrolled = clock();
    double duration_sse_unrolled = static_cast<double>(stop_sse_unrolled - start_sse_unrolled) / CLOCKS_PER_SEC;
    double avg_duration_sse_unrolled = duration_sse_unrolled / num_repeats * 1e6; // convert to microseconds

    // measure SSE-based matrix multiplication with half unrolled loop time
    clock_t start_sse_unrolled_extreme = clock();
    for (int i = 0; i < num_repeats; ++i) {
        matrixMultiplySSEUnrolledExtreme(A, B, C_sse_unrolled_extreme, size);
    }
    clock_t stop_sse_unrolled_extreme = clock();
    double duration_sse_unrolled_extreme = static_cast<double>(stop_sse_unrolled_extreme - start_sse_unrolled_extreme) / CLOCKS_PER_SEC;
    double avg_duration_sse_unrolled_extreme = duration_sse_unrolled_extreme / num_repeats * 1e6; // convert to microseconds

    // measure non-SSE matrix multiplication time
    clock_t start_non_sse = clock();
    for (int i = 0; i < num_repeats; ++i) {
        matrixMultiply(A, B, C_non_sse, size);
    }
    clock_t stop_non_sse = clock();
    double duration_non_sse = static_cast<double>(stop_non_sse - start_non_sse) / CLOCKS_PER_SEC;
    double avg_duration_non_sse = duration_non_sse / num_repeats * 1e6; // convert to microseconds

    // print the results
    std::cout << "Matrix A:\n";
    printMatrix(A, size, size);

    std::cout << "Matrix B:\n";
    printMatrix(B, size, size);

    std::cout << "SSE-based Matrix Multiplication Result:\n";
    printMatrix(C_sse, size, size);
    std::cout << "Average time taken by SSE-based matrix multiplication: " << avg_duration_sse << " microseconds\n\n\n";

    std::cout << "SSE-based Matrix Multiplication (Unrolled) Result:\n";
    printMatrix(C_sse_unrolled, size, size);
    std::cout << "Average time taken by SSE-based matrix multiplication (Unrolled): " << avg_duration_sse_unrolled << " microseconds\n\n\n";

    std::cout << "SSE-based Matrix Multiplication (Unrolled Extreme) Result:\n";
    printMatrix(C_sse_unrolled_extreme, size, size);
    std::cout << "Average time taken by SSE-based matrix multiplication (Unrolled Extreme): " << avg_duration_sse_unrolled_extreme << " microseconds\n\n\n";

    std::cout << "Non-SSE Matrix Multiplication Result:\n";
    printMatrix(C_non_sse, size, size);
    std::cout << "Average time taken by non-SSE matrix multiplication: " << avg_duration_non_sse << " microseconds\n\n\n";

    return 0;
}

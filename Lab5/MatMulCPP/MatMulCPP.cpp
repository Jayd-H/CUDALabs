#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    const int heightA = 4;
    const int widthA = 3;
    const int heightB = 3;
    const int widthB = 2;

    const int arraySizeA = heightA * widthA;
    const int arraySizeB = heightB * widthB;
    const int arraySizeC = heightA * widthB;

    const int a[arraySizeA] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    const int b[arraySizeB] = {
        1, 2,
        3, 4,
        5, 6
    };

    int c[arraySizeC] = { 0 };

	// this is worthless right now because the matrices are so small its less than a microsecond, its here for future reference
	auto start = std::chrono::high_resolution_clock::now();

    // actual matrix multiplication 
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            int sum = 0;
            for (int k = 0; k < widthA; k++) {
                // A[i][k] * B[k][j]
                sum += a[i * widthA + k] * b[k * widthB + j];
            }
            c[i * widthB + j] = sum;
        }
    }

	auto end = std::chrono::high_resolution_clock::now();

	// these are just ways to print the matrices

    std::cout << "Matrix A (" << heightA << "x" << widthA << "):" << std::endl;
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthA; j++) {
            std::cout << std::setw(4) << a[i * widthA + j];
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B (" << heightB << "x" << widthB << "):" << std::endl;
    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
            std::cout << std::setw(4) << b[i * widthB + j];
        }
        std::cout << std::endl;
    }

    std::cout << "\nResult Matrix C = A * B (" << heightA << "x" << widthB << "):" << std::endl;
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            std::cout << std::setw(6) << c[i * widthB + j];
        }
        std::cout << std::endl;
    }

	std::cout << "\nTime taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    return 0;
}
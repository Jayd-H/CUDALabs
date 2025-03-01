#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

// matrix dimensions
const int heightA = 4;
const int widthA = 3;
const int heightB = 3;
const int widthB = 2;
const int arraySizeA = heightA * widthA;
const int arraySizeB = heightB * widthB;
const int arraySizeC = heightA * widthB;

// each thread computes one element of C
__global__ void matrixMulKernel(int* c, const int* a, const int* b)
{
    // thread index within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // check if this thread should compute a result element
    if (row < heightA && col < widthB) {
        int sum = 0;
        // dot product loop
        for (int k = 0; k < widthA; k++) {
            sum += a[row * widthA + k] * b[k * widthB + col];
        }
        c[row * widthB + col] = sum;
    }
}

// helper function for using cuda to perform matrix multiplication
cudaError_t matrixMulWithCuda(int* c, const int* a, const int* b);

int main()
{
    // define input matrices
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

    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = matrixMulWithCuda(c, a, b);
    auto end = std::chrono::high_resolution_clock::now();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMulWithCuda failed!");
        return 1;
    }

    // fancy way to print the matrices
    std::cout << "Matrix A (" << heightA << "x" << widthA << "):" << std::endl;
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthA; j++) {
            std::cout << a[i * widthA + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B (" << heightB << "x" << widthB << "):" << std::endl;
    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
            std::cout << b[i * widthB + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nResult Matrix C = A * B (" << heightA << "x" << widthB << "):" << std::endl;
    for (int i = 0; i < heightA; i++) {
        for (int j = 0; j < widthB; j++) {
            std::cout << c[i * widthB + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nCUDA Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;

    // reset cuda device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t matrixMulWithCuda(int* c, const int* a, const int* b)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;


    cudaStatus = cudaMalloc((void**)&dev_c, arraySizeC * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arraySizeA * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySizeB * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, arraySizeA * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySizeB * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threadsPerBlock(widthB, heightA);
    matrixMulKernel << <1, threadsPerBlock >> > (dev_c, dev_a, dev_b);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixMulKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, arraySizeC * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

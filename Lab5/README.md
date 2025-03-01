# Lab 5

## Exercise 1

### CPU C++

I started with the CPU only C++ solution, using nested for loops to iterate through the rows and columns of the matricies. I believe this is the standard way of doing this. 

```cpp

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

	// these are just cool ways to print the matrices

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

```

Here is the result:

```

Matrix A (4x3):
   1   2   3
   4   5   6
   7   8   9
  10  11  12

Matrix B (3x2):
   1   2
   3   4
   5   6

Result Matrix C = A * B (4x2):
    22    28
    49    64
    76   100
   103   136

Time taken: 0 microseconds

```

### GPU CUDA - Single Thread Block 

After re-familiarising that process in pure C++, it is time for the CUDA implementation. A lot of the code here is identical if not very similar to the C++ solution, understandably. This might be a bit cheeky but the introduction to the cuda-samples samples has a lot of code related to matrix multiplication in CUDA. It makes sense and I used that to better understand the code here, taking liberally where I can, so I am quite glad I found it. I opted to store the matrices as 1D arrays in row-major order, calculating the appropriate indices manually because of my familiarity with my C++ implementation.  

[matrixMul - Cuda Samples Introduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/matrixMul)

```cpp

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


```

The main difference here is obviously all the extra work we have to do simply communicating with the GPU in this way. A single 2D thread block is used matching the output matrix (widthB * heightA). Each thread here is responsible for computing exactly one element of the result matrix. The way each thread knows which index to compute is specified in the kernel, with each one calculating the dot product of a row from matrix A and a column from matrix B. 

The primary limitation of this approach, like the lab sheet says, is scalability. Because CUDA hardware limits the max number of threads per block, this implementation can only handle small matricies. Therefore, for larger matricies, this approach would fail because we would need more threads than can fit in a single block. 

```

Matrix A (4x3):
1 2 3
4 5 6
7 8 9
10 11 12

Matrix B (3x2):
1 2
3 4
5 6

Result Matrix C = A * B (4x2):
22 28
49 64
76 100
103 136

CUDA Time taken: 93631 microseconds

```

The reported time here seems exceptionally high for such a small matrix multiplication. My theory is that this time likely includes the overhead of CUDA initialisation, memory allocation, and data transfers, not just purely computation time. 

## Lab 5 Reflection

This lab was a lot more chill than the previous ones I think, but that may be because I am getting the hang of things more in CUDA than I did before. I am very glad I had a look through the CUDA samples prior to this and remembered it talk about matrix multiplication extensively. I have to admit, a lot of the CUDA samples there do seem a bit daunting to me still, but I am sure I will get the hang of them in a few weeks. I liked this lab because it showed the limitations of using CUDA too. I did actually implement a version with multiple thread blocks and managed to get better results (by around 300-400 microseconds) but I ommitted it here for brevity as it was optional. I am excited to build upon this lab, as matrix multiplication is (what I imagine) the core of so much to come.
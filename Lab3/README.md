# Lab 3

## Q1 Vector Dot-Product

### CPU Solution

```cpp

int CPUDotProduct(const int* a, const int* b, int size) {
    int result = 0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

int main(){
    const int a[5] = { 1, 2, 3, 4, 5 };
    const int b[5] = { 10, 10, 10, 10, 10 };

    // Q1
    int cpud = CPUDotProduct(a, b, 5);

    printf("CPU dot product = %d\n", cpud);
}

```

I started with a simple class to calculate it, then printing the final value in main, giving the result of 150. This one was fairly straight-forward compared to using the GPU.

### GPU Solution

```cpp

__global__ void multiplyKernel(int* c, const int* a, const int* b){
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int GPUDotProduct(const int* a, const int* b, int size) {
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    int* c = new int[size];

    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // *this line keeps throwing an error but its incorrect
    multiplyKernel << <1, size >> > (dev_c, dev_a, dev_b);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    int dotProduct = 0;
    for (int i = 0; i < size; i++) {
        dotProduct += c[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    delete[] c;

    return dotProduct;
}

int main(){
    const int a[5] = { 1, 2, 3, 4, 5 };
    const int b[5] = { 10, 10, 10, 10, 10 };

    // Q1
    int cpud = CPUDotProduct(a, b, 5);
    int gpud = GPUDotProduct(a, b, 5);

    printf("CPU dot product = %d\n", cpud);
    printf("GPU dot product = %d\n", gpud);
}

```

I made a new kernel, `multiplyKernel` for this, that simple off-loads the workload of multiplying two indexes together on its own unique thread. For this, we have to ensure we are allocating and then freeing the right amount of memory.

## Q2 Vector Dot-Product using Unified Memory

### Managed Memory

Theres two types of way to use managed unified memory, dynamic managed memory, and just managed memory. Managed memory is static, where the memory is globally declared with a fixed size as size must be known at compile time.

```cpp

__device__ __managed__ int managed_a[5];
__device__ __managed__ int managed_b[5];
__device__ __managed__ int managed_c[5];

int MMDotProduct(const int* a, const int* b, int size) {

    for (int i = 0; i < size; i++) {
        managed_a[i] = a[i];
        managed_b[i] = b[i];
    }

    PerElement_AtimesB << <1, size >> > (managed_c, managed_a, managed_b);
    cudaDeviceSynchronize();

    int dotProduct = 0;
    for (int i = 0; i < size; i++) {
        dotProduct += managed_c[i];
    }

    return dotProduct;
}

int main(){
    const int a[5] = { 1, 2, 3, 4, 5 };
    const int b[5] = { 10, 10, 10, 10, 10 };

    // Q1
    int cpud = CPUDotProduct(a, b, 5);
    int gpud = GPUDotProduct(a, b, 5);

    printf("CPU dot product = %d\n", cpud);
    printf("GPU dot product = %d\n", gpud);

    // Q2
    int dmmd = DMMDotProduct(a, b, 5);

    printf("managed memory dot product = %d\n", mmd);

    return 0;
}
```

### Dynamic Managed Memory

Dynamic managed memory instead uses `cudaMallocManaged()` to dynamically allocate memory at runtime, it is allocated and freed within the function, allowing it to be more flexible. Furthermore, it also leads to less lines of code.

```cpp

int DMMDotProduct(const int* input_a, const int* input_b, int size) {
    int* a, * b, * c;

    // allocated unified memory accessible by both gpu and cpu
    cudaMallocManaged(&a, size * sizeof(int));
    cudaMallocManaged(&b, size * sizeof(int));
    cudaMallocManaged(&c, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        a[i] = input_a[i];
        b[i] = input_b[i];
    }

    PerElement_AtimesB << <1, size >> > (c, a, b);
    cudaDeviceSynchronize();

    int dotProduct = 0;
    for (int i = 0; i < size; i++) {
        dotProduct += c[i];
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return dotProduct;
}

int main(){
    const int a[5] = { 1, 2, 3, 4, 5 };
    const int b[5] = { 10, 10, 10, 10, 10 };

    // Q1
    int cpud = CPUDotProduct(a, b, 5);
    int gpud = GPUDotProduct(a, b, 5);

    printf("CPU dot product = %d\n", cpud);
    printf("GPU dot product = %d\n", gpud);

    // Q2
    int dmmd = DMMDotProduct(a, b, 5);
    int mmd = MMDotProduct(a, b, 5);

    printf("dynamic managed memory dot product = %d\n", dmmd);
    printf("managed memory dot product = %d\n", mmd);

    return 0;
}

```

# CUDA LAB 2

## Exercise 1

```addKernel << <1, 5 >> > (dev_c, dev_a, dev_b);```: 1 group of 5 threads.

```addKernel << <2, 3 >> > (dev_c, dev_a, dev_b);```: 2 groups of 3 threads each.

```addKernel << <3, 2 >> > (dev_c, dev_a, dev_b);```: 3 groups of 2 threads each.

```addKernel << <6, 1 >> > (dev_c, dev_a, dev_b);```: 6 groups of 1 thread each. 

## Exercise 2

Changing this line ```int i = threadIdx.x;``` to ```int i = threadIdx.x + blockIdx.x * blockDim.x;``` ensures that every index for each thread regardless of block has a unique ID.

With this change, it then allows us to assign a 3 groups of 2 threads with the line ```addKernel << <3, 2 >> > (dev_c, dev_a, dev_b);```, adding the to matricies ```const int a[arraySize] = { 1, 2, 3 };``` and ```const int b[arraySize] = { 10, 20, 30 };```, getting us a result of ```{11, 22, 33}```. 

For this, I had to butcher up the template a bit though.

```cpp
int main()
{
    const int arraySize = 3;
    const int a[arraySize] = { 1, 2, 3 };
    const int b[arraySize] = { 10, 20, 30 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf(" = {% d,% d,% d}\n",
        c[0], c[1], c[2], c[3]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
```
```cpp
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    //* Previous code ommitted for breity. 
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<2, size>>>(dev_c, dev_a, dev_b);
}

```


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

## Exercise 3

The difference between these Kernels and the ones previously is that it creates a 2D block of threads. Therefore each thread has an X and Y coordinate. 

```addKernel << <1, dim3(2, 3) >> > (dev_c, dev_a, dev_b);``` creates a 2D 2x3 block of threads. 2 rows, 3 columns.

```addKernel << <1, dim3(3, 3) >> > (dev_c, dev_a, dev_b);``` creates a 2D 3x3 block of threads.

## Exercise 4

```int i = threadIdx.x;``` is not sufficient index handling for threads when creating 2D blocks with ```addKernel << <1, dim3(3, 2) >> > (dev_c, dev_a, dev_b);```. Using this, threads do not get unique ID's in their 2D block as they dont have an X and Y coordinate.

We would need to change this code to something like this: ```int i = threadIdx.x + threadIdx.y * blockDim.x;```. This means for ```addKernel << <1, dim3(3, 2) >> > (dev_c, dev_a, dev_b);``` it would be a 2D 3x2 block of threads. 

## Exercise 5

```addKernel<<<dim3(2,3), dim3(2,2)>>>(dev_c, dev_a, dev_b);``` is more complex as it is a Kernel with a 2x3 group of blocks, with each block containing a 2x2 group of threads. This means that the previous solution will not work here similarly to how the solution for exercise 2 did not work when groups of threads were introduced in exercise 3. 

To ensure each thread has a unique ID, we will need to assign them something like this:

```cpp
int i = (threadIdx.x + blockIdx.x * blockDim.x) + 
        (threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x);
```

This makes sure that each thread ends up with a unique index throughout the group of blocks and the group of threads. 

## Reflection

I enjoyed the more theory-heavy side of this lab rather than it being very code heavy. It deepened my knowledge on how groups of blocks and threads work, especially when running into the issue of duplicate thread ID's. Not mentioned in this lab, but through working through it, it allowed me to revisit material on why the GPU works with blocks like this, revisiting my knowledge on warps and whatnot. I know I have a long way to go in my CUDA journey but I am appreciating this start and am excited to see this knowledge and compartmentalisation of threads in practice. 




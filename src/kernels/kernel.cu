//#include "../../include/kernel.cuh"
#include "../../include/template.cuh"

//#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void addKernel(const int* a, const int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size)
    {
        c[index] = a[index] + b[index];
    }
}

void vectorAdd(
    const int* a,
    const int* b,
    int* c,
    int size
)
{
    int* dev_a;
    int* dev_b;
    int* dev_c;


    // Allocate GPU memory
    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));


    // Copy CPU -> GPU
    cudaMemcpy(
        dev_a,
        a,
        size * sizeof(int),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        dev_b,
        b,
        size * sizeof(int),
        cudaMemcpyHostToDevice
    );


    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;


    ExecuteKernel("add kernel", blocks, threads, 0, 0, addKernel, dev_a, dev_b, dev_c, size);


    cudaDeviceSynchronize();


    // Copy GPU -> CPU
    cudaMemcpy(
        c,
        dev_c,
        size * sizeof(int),
        cudaMemcpyDeviceToHost
    );


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
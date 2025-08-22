#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "../Utils/Macros.hpp"
#include "../nvtx3/nvtx3.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h> 
#include <type_traits>
#include <curand_kernel.h>
#include <iostream>

_AM_START

template<typename KernelFunc, typename... Args>
cudaError_t ExecuteKernel(const char* label, dim3 blocks, dim3 threads, size_t sharedMem, cudaStream_t stream, KernelFunc Kernel, Args&&... args) {

    nvtx3::scoped_range range{ label };
    Kernel<<<blocks, threads, sharedMem, stream>>>(std::forward<Args>(args)...);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return err;
    }

    return err;
}


_KERNELS_START

template<typename T>
__global__ void FillTensor(T* data, T value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

template<typename T>
__global__ void FillTensorRandom(T* data, size_t size, T llimit, T rlimit, unsigned long long seed) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float rnd = curand_uniform(&state);

    if constexpr (std::is_same<T, __half>::value) {
        __half l = llimit;
        __half r = rlimit;
        __half range = __hsub(r, l);
        __half scaled = __hmul(__float2half(rnd), range);
        data[idx] = __hadd(l, scaled);
    }
    else {
        data[idx] = static_cast<T>(llimit + rnd * (rlimit - llimit));
    }
}

_KERNELS_END


_AM_END

#endif// Kernels.cuh
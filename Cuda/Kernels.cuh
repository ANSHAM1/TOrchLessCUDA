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
#include <string_view>
#include <algorithm>

template<size_t N>
struct FixedString {
    char data[N]{};

    constexpr FixedString(const char(&str)[N]) {
        std::copy_n(str, N, data);
    }

    constexpr operator std::string_view() const {
        return { data, N - 1 };
    }
};

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


template<typename T, FixedString type>
__global__ void ActivationKernel(const T* Input, T* Output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const T value = Input[idx];

    if constexpr (type == "relu") {
        if constexpr (std::is_same_v<T, __half>)
            Output[idx] = __hgt(value, __float2half(0.0f)) ? value : __float2half(0.0f);
        else 
            Output[idx] = max(value, T(0));
    }
    else if constexpr (type == "tanh") {
        if constexpr (std::is_same_v<T, __half>) {
            float val_f = __half2float(value);
            Output[idx] = __float2half(tanhf(val_f));
        }
        else if constexpr (std::is_same_v<T, float>)
            Output[idx] = tanhf(value);
        else
            Output[idx] = tanh(value);
    }
    else if constexpr (type == "sigmoid") {
        if constexpr (std::is_same_v<T, __half>) {
            float val_f = __half2float(value);
            Output[idx] = __float2half(1.0f / (1.0f + expf(-val_f)));
        }
        else if constexpr (std::is_same_v<T, float>)
            Output[idx] = 1.0f / (1.0f + expf(-value));
        else
            Output[idx] = 1.0 / (1.0 + exp(-value));
    }
    else
        static_assert(false, "Unsupported activation type provided to kernel.");
}

// ---------------------------------------------------------------------------------------------------------------------

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
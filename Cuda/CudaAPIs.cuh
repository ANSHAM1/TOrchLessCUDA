#ifndef __CUDA_API_CUH__
#define __CUDA_API_CUH__

#include "../Utils/Macros.hpp"

#include "cuda_runtime.h"
#include <curand.h>
#include <type_traits>
#include <cuda_fp16.h> 

__global__ void FloatToHalf(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

_CUDNN_START

template<typename T>
curandStatus_t FillTensorRandom(T* data, size_t size, cudaStream_t stream, unsigned long long seed) {
    curandGenerator_t gen;
    curandStatus_t status;

    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) return status;

    status = curandSetStream(gen, stream);
    if (status != CURAND_STATUS_SUCCESS) return status;

    status = curandSetPseudoRandomGeneratorSeed(gen, seed);
    if (status != CURAND_STATUS_SUCCESS) return status;

    if constexpr (std::is_same_v<T, float>) {
        status = curandGenerateUniform(gen, data, size);
    }
    else if constexpr (std::is_same_v<T, double>) {
        status = curandGenerateUniformDouble(gen, data, size);
    }
    else if constexpr (std::is_same_v<T, __half>) {
        float* tmp;
        cudaMallocAsync(&tmp, size * sizeof(float), stream);
        status = curandGenerateUniform(gen, tmp, size);

        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        FloatToHalf<<<blocks, threads, 0, stream>>>(tmp, data, size);
        cudaFreeAsync(tmp, stream);
    }
    else {
        static_assert(std::is_same_v<T, float> ||
            std::is_same_v<T, double> ||
            std::is_same_v<T, __half>,
            "FillTensorRandom only supports float, double, or __half.");
    }

    curandDestroyGenerator(gen);
    return status;
}

_CUDNN_END

#endif// CudaAPIs.cuh
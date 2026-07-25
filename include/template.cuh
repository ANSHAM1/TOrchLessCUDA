#pragma once

#include "cuda_runtime.h"




template<typename KernelFunc, typename... Args>
cudaError_t ExecuteKernel(const char* label, dim3 blocks, dim3 threads, size_t sharedMem,
    cudaStream_t stream, KernelFunc Kernel, Args&&... args) {

    Kernel << < blocks, threads, sharedMem, stream >> > (std::forward<Args>(args)...);

    return cudaGetLastError();
}
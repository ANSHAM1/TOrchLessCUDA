#ifndef __MACROS_HPP__
#define __MACROS_HPP__


#ifdef __CUDACC__
#define IF_CUDA_AVAILABLE(code) code
#else
#define IF_CUDA_AVAILABLE(code)
#endif


#ifdef USE_CUDA
#define CUDA_CHECK(call) do {                              \
    cudaError_t _e = (call);                               \
    if (_e != cudaSuccess) {                               \
        std::string s = std::string("CUDA error: ") +      \
            cudaGetErrorString(_e);                        \
        throw std::runtime_error(s);                       \
    }                                                      \
} while(0)
#else
#define CUDA_CHECK(call) do { (void)call; } while(0)
#endif


#endif// Macros.hpp
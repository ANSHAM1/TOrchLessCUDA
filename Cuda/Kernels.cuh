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

// ==========================================================================================
//  Optimized Tiled 2D Convolution Kernel
// ==========================================================================================

//CUDA Grid(Thousands of Blocks)
//+ -----------------------------------------------------+
//|                                                      |
//|   Grid Dimension Z : Maps to(Batch, Output Channel)  |
//|   Grid Dimension Y : Maps to Output Height Tiles     |
//|   Grid Dimension X : Maps to Output Width Tiles      |
//|                                                      |
//+------------------------------------------------------+
//|
//| Each cube in this grid is one Thread Block.
//| It is assigned a unique(blockIdx.x, blockIdx.y, blockIdx.z).
//|
//V
//One Thread Block(e.g., 128 threads)
//+ -----------------------------------------------------+
//|                                                      |
//|   Responsibility: Compute one 16x16 output tile.     |
//|                                                      |
//|   blockIdx.z->Determines which image in the batch    |
//| and which output channel(filter)                     |
//|                 this block works on.                 |
//|                                                      |
//|   blockIdx.y->Determines the row of the tile in      |
//|                 the output feature map.              |
//|                                                      |
//|   blockIdx.x->Determines the column of the tile.     |
//|                                                      |
//+------------------------------------------------------+
//|
//| The block achieves this using a 3 - step process
//| involving shared memory.
//|
//V
//+ -----------------------+ -- +-------------------------+
//| GLOBAL MEMORY(Slow)    |    |   GLOBAL MEMORY(Slow)   |
//|      Input Tensor      |    |      Kernel Tensor      |
//+------------------------+ -- +-------------------------+
//^ ^
//| (Step 1: Coalesced Load)         |
//|                                  |
//+-------------------------------------------------------+
//| SHARED MEMORY(Fast Cache)                             |
//|  +-------------------+ - +------------------------+   |
//|  |   Input Tile      |   |   Kernel Tile          |   |
//|  +-------------------+ - +------------------------+   |
//+-------------------------------------------------------+
//^ ^
//| (Step 3: Fast Reads)                                  |
//|                                                       |
//+----------------------------------------------------------+
//| THREADS(Registers - Fastest)                             |
//|                                                          |
//|   Thread(tx, ty) :                                       |
//|   -Loads one piece of Input & Kernel into Shared Mem.    |
//|   -Accumulates result for one output pixel in a register.|
//|                                                          |
//+----------------------------------------------------------+
//|
//| (Step 4: Final Write)
//V
//+ ------------------------+
//| GLOBAL MEMORY(Slow)     |
//|      Output Tensor      |
//+-------------------------+

// REMINDER: This "tiled" approach is much faster than a naive kernel because it uses
// fast shared memory as a cache to minimize slow global memory access.

// TEMPLATE PARAMS: These must be compile-time constants. This allows the compiler
// to perform major optimizations like loop unrolling.
template<typename T, int TILE_DIM, int BLOCK_ROWS, int KERNEL_TILE_DIM>
__global__ void TiledConv2dKernelF(const T* input, const T* kernel, T* output,
    const int N, const int C, const int H, const int W,
    const int K, const int KH, const int KW,
    const int S, const int P,
    const int OH, const int OW) {

    // --- Shared Memory: A fast, programmable L1 cache for one thread block ---
    // Used to store a "tile" of the input and kernel, avoiding repeated global memory reads.
    constexpr int PADDED_TILE_DIM = (TILE_DIM - 1) * S + KERNEL_TILE_DIM;
    extern __shared__ T s_data[];
    T* s_input = s_data;
    T* s_kernel = &s_data[PADDED_TILE_DIM * PADDED_TILE_DIM];

    // --- Indexing: Map threads and blocks to the overall problem ---
    // A thread block computes one `TILE_DIM x TILE_DIM` tile of the output.
    const int tile_x = blockIdx.x * TILE_DIM;
    const int tile_y = blockIdx.y * TILE_DIM;
    // A thread computes one pixel within that tile.
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;
    const int out_x = tile_x + thread_x;
    const int out_y = tile_y + thread_y;
    // The Z-dimension of the grid maps to the batch and output channel.
    const int batch_idx = blockIdx.z / K;
    const int kernel_idx = blockIdx.z % K;

    // Use a register for the accumulator: fastest possible memory.
    T accumulator = T(0);

    // --- Main Loop: Process one input channel at a time ---
    for (int c = 0; c < C; ++c) {
        // --- Step 1: Cooperative & Coalesced Load ---
        // The block's threads work as a team to load data from slow global memory
        // into fast shared memory. This is done in a structured ("coalesced") way
        // to maximize memory bandwidth.
        int input_tile_start_y = tile_y * S - P;
        int input_tile_start_x = tile_x * S - P;
        for (int i = thread_y; i < PADDED_TILE_DIM; i += BLOCK_ROWS) {
            for (int j = thread_x; j < PADDED_TILE_DIM; j += TILE_DIM) {
                int load_y = input_tile_start_y + i;
                int load_x = input_tile_start_x + j;
                if (load_y >= 0 && load_y < H && load_x >= 0 && load_x < W) {
                    s_input[i * PADDED_TILE_DIM + j] = input[((batch_idx * C + c) * H + load_y) * W + load_x];
                }
                else {
                    s_input[i * PADDED_TILE_DIM + j] = T(0); // Handle padding
                }
            }
        }
        if (thread_y < KERNEL_TILE_DIM && thread_x < KERNEL_TILE_DIM) {
            s_kernel[thread_y * KERNEL_TILE_DIM + thread_x] = kernel[((kernel_idx * C + c) * KH + thread_y) * KW + thread_x];
        }

        // --- Step 2: Synchronize ---
        // Wait until ALL threads in the block have finished loading into shared memory.
        __syncthreads();

        // --- Step 3: Local Computation ---
        // Each thread computes its result using ONLY the data in fast shared memory.
        // This avoids the global memory bottleneck and is the key to performance.
        if (out_y < OH && out_x < OW) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    accumulator += s_input[(thread_y * S + kh) * PADDED_TILE_DIM + (thread_x * S + kw)] * s_kernel[kh * KW + kw];
                }
            }
        }

        // --- Step 4: Synchronize Again ---
        // Wait for all calculations to finish before the next loop iteration overwrites shared memory.
        __syncthreads();
    }

    // --- Step 5: Final Write ---
    // Each thread writes its final result to global memory only ONCE.
    if (out_y < OH && out_x < OW) {
        output[((batch_idx * K + kernel_idx) * OH + out_y) * OW + out_x] = accumulator;
    }
}

// -----------------------------------------------------------------------------------------------------------------------

template<typename T, FixedString type>
__global__ void ActivationKernelF(const T* Input, T* Output, size_t size) {
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

// -----------------------------------------------------------------------------------------------------------------------

template<typename T>
__global__ void MaxPool2DKernelF(
    const T* __restrict__ Input,
    T* __restrict__ Output,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int kH, int kW,
    int Stride, int Padding)
{
    extern __shared__ T tile[];

    int n = blockIdx.z;     // batch index
    int c = blockIdx.y;     // channel index

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = threadIdx.y;

    if (out_x >= Wout || out_y >= Hout) return;

    // Map to input coordinates
    int in_x_start = out_x * Stride - Padding;
    int in_y_start = out_y * Stride - Padding;

    T maxval = -FLT_MAX;

    for (int ky = 0; ky < kH; ky++) {
        for (int kx = 0; kx < kW; kx++) {
            int in_y = in_y_start + ky;
            int in_x = in_x_start + kx;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                size_t idx = ((n * C + c) * H + in_y) * W + in_x;
                T v = Input[idx];
                maxval = v > maxval ? v : maxval;
            }
        }
    }

    size_t out_idx = ((n * C + c) * Hout + out_y) * Wout + out_x;
    Output[out_idx] = maxval;
}

template<typename T>
__global__ void AvgPool2DKernelF(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int out_row = blockIdx.x / outW;
    int out_col = blockIdx.x % outW;

    if (out_row >= outH || out_col >= outW) return;

    int h_start = out_row * strideH - padH;
    int w_start = out_col * strideW - padW;

    T sum = T(0);
    int count = 0;

    for (int i = 0; i < kH; ++i) {
        int h = h_start + i;
        if (h >= 0 && h < H) {
            for (int j = 0; j < kW; ++j) {
                int w = w_start + j;
                if (w >= 0 && w < W) {
                    int input_idx = ((n * C + c) * H + h) * W + w;
                    sum += input[input_idx];
                    count++;
                }
            }
        }
    }

    int output_idx = ((n * C + c) * outH + out_row) * outW + out_col;
    output[output_idx] = count > 0 ? sum / count : T(0);
}

// -----------------------------------------------------------------------------------------------------------------------

template<typename T>
__global__ void DenseKernelF(
    const T* __restrict__ input,   // [batch_size, InNumFeature]
    const T* __restrict__ weight,  // [OutNumFeature, InNumFeature]
    const T* __restrict__ bias,    // [OutNumFeature]
    T* __restrict__ output,        // [batch_size, OutNumFeature]
    int batch_size,
    int InNumFeature,
    int OutNumFeature)
{
    // Each thread computes one output element (b, o)
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_idx >= OutNumFeature)
        return;

    T sum = 0;

    // Iterate over all input features
    for (int in_idx = 0; in_idx < InNumFeature; ++in_idx) {
        T in_val = input[batch_idx * InNumFeature + in_idx];
        T w_val = weight[out_idx * InNumFeature + in_idx];
        sum += in_val * w_val;
    }

    // Add bias
    sum += bias[out_idx];

    // Store result
    output[batch_idx * OutNumFeature + out_idx] = sum;
}

// -----------------------------------------------------------------------------------------------------------------------

template<typename T>
__global__ void SigmoidKernelF(
    const T* __restrict__ input,   // [batch_size, 1, 1, classDim]
    T* __restrict__ output,        // [batch_size, 1, 1, classDim]
    int total_elements)           // batch_size * classDim
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    T x = input[idx];
    output[idx] = static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

template<typename T>
__global__ void SoftmaxKernelF(
    const T* __restrict__ input,   // [batch_size, 1, 1, classDim]
    T* __restrict__ output,        // [batch_size, 1, 1, classDim]
    int batch_size,
    int classDim)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int base_idx = b * classDim;

    // Step 1: Find max value for numerical stability
    T max_val = input[base_idx];
    for (int i = 1; i < classDim; ++i) {
        max_val = max(max_val, input[base_idx + i]);
    }

    // Step 2: Compute exponentials and sum
    T sum_exp = static_cast<T>(0);
    for (int i = 0; i < classDim; ++i) {
        output[base_idx + i] = exp(input[base_idx + i] - max_val);
        sum_exp += output[base_idx + i];
    }

    // Step 3: Normalize to get softmax output
    for (int i = 0; i < classDim; ++i) {
        output[base_idx + i] /= sum_exp;
    }
}

template<typename T>
__global__ void SigmoidBCEKernelF(
    const T* __restrict__ input,   // [batch_size, 1, 1, 2]
    const T* __restrict__ label,   // [batch_size, 1, 1, 2]
    T* __restrict__ output,        // [batch_size, 1, 1, 2]
    T* __restrict__ loss,          // [batch_size]
    int batch_size)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // For simplicity assume classDim == 2 for sigmoid output
    int base_idx = b * 2;

    // Apply sigmoid activation
    T x0 = input[base_idx];
    T x1 = input[base_idx + 1];

    T s0 = static_cast<T>(1) / (static_cast<T>(1) + exp(-x0));
    T s1 = static_cast<T>(1) / (static_cast<T>(1) + exp(-x1));

    output[base_idx] = s0;
    output[base_idx + 1] = s1;

    // Compute binary cross-entropy loss
    T y0 = label[base_idx];
    T y1 = label[base_idx + 1];

    T l0 = -(y0 * log(max(s0, static_cast<T>(1e-7))) + (static_cast<T>(1) - y0) * log(max(static_cast<T>(1) - s0, static_cast<T>(1e-7))));
    T l1 = -(y1 * log(max(s1, static_cast<T>(1e-7))) + (static_cast<T>(1) - y1) * log(max(static_cast<T>(1) - s1, static_cast<T>(1e-7))));

    loss[b] = l0 + l1;
}

template<typename T>
__global__ void SoftmaxCCEKernelF(
    const T* __restrict__ input,   // [batch_size, 1, 1, classDim]
    const T* __restrict__ label,   // [batch_size, 1, 1, classDim]
    T* __restrict__ output,        // [batch_size, 1, 1, classDim]
    T* __restrict__ loss,          // [batch_size]
    int batch_size,
    int classDim)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int base_idx = b * classDim;

    // 1. Find max for numerical stability
    T max_val = input[base_idx];
    for (int i = 1; i < classDim; ++i) {
        max_val = max(max_val, input[base_idx + i]);
    }

    // 2. Compute exponentials and sum
    T sum_exp = static_cast<T>(0);
    for (int i = 0; i < classDim; ++i) {
        output[base_idx + i] = exp(input[base_idx + i] - max_val);
        sum_exp += output[base_idx + i];
    }

    // 3. Normalize to get softmax output
    for (int i = 0; i < classDim; ++i) {
        output[base_idx + i] /= sum_exp;
    }

    // 4. Compute categorical cross-entropy loss
    T l = static_cast<T>(0);
    for (int i = 0; i < classDim; ++i) {
        T y = label[base_idx + i];
        T s = max(output[base_idx + i], static_cast<T>(1e-7)); // prevent log(0)
        l += -y * log(s);
    }
    loss[b] = l;
}

// -----------------------------------------------------------------------------------------------------------------------

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

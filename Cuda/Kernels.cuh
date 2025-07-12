#ifndef KERNELS_CUH
#define KERNELS_CUH

constexpr auto TILE_SIZE = 16;

#include "BatchDevice.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cfloat>
#include <ctime>
using namespace std;

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(code)
			<< " at " << file << ":" << line << std::endl;
		if (abort) exit(code);
	}
}

// Random Generator
__global__ void initCurand(unsigned int seed, curandState* states, int size);

// Activation
__global__ void Relu(BatchDevice* BWout, BatchDevice* BWi);
__global__ void Sigmoid(BatchDevice* BWout, BatchDevice* BWi);
__global__ void Tanh(BatchDevice* BWout, BatchDevice* BWi);

// DeActivation
__global__ void dRelu(BatchDevice* dBWi, BatchDevice* dBWout);
__global__ void dSigmoid(BatchDevice* dBWi, BatchDevice* dBWout);
__global__ void dTanh(BatchDevice* dBWi, BatchDevice* dBWout);

// Convolutuion 2D
__global__ void conv2dKernelF(BatchDevice* output, BatchDevice* inputs, BatchDevice* kernels, int S, int P);
__global__ void conv2dKernelB(BatchDevice* input, BatchDevice* output, BatchDevice* kernels, int S, int P, float L, float R2);
__global__ void kernelsUpdate(BatchDevice* kernels, float lr);

// Poolings
__global__ void MaxPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);
__global__ void AvgPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);
__global__ void MinPoolKernelF(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);

__global__ void MaxPoolKernelB(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);
__global__ void AvgPoolKernelB(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);
__global__ void MinPoolKernelB(BatchDevice* output, BatchDevice* inputs, int kH, int kW, int S, int P);

// Batch Normalization
__global__ void devideKernelF(float* sum, int num_per_channel, int C);
__global__ void MeanSumKernelF(BatchDevice* inputs, float* x_sum);
__global__ void VarSumKernelF(BatchDevice* inputs, float* var_sum, float* x_);
__global__ void BatchNormKernelF(BatchDevice* output, BatchDevice* inputs, float* var_, float* x_, BatchDevice* gamma, BatchDevice* beta);

__global__ void dGammaBetaKernelB(BatchDevice* dY, BatchDevice* X, float* mean, float* var, BatchDevice* gamma, BatchDevice* beta, float L, float R2);
__global__ void SumDyKernelB(float* sum_dy, float* sum_dy_xhat, BatchDevice* outputs, BatchDevice* inputs, float* mean, float* var);
__global__ void BatchNormKernelB(BatchDevice* inputs, BatchDevice* outputs, float* mean, float* var, BatchDevice* gamma, float* sum_dy, float* sum_dy_xhat, int C, int N, float L, float R2);
__global__ void gammabetaUpdate(BatchDevice* gamma, BatchDevice* beta, float lr);

// Dropout 
__global__ void DropoutKernelF(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* mask, curandState* states, float p);
__global__ void DropoutKernelB(BatchDevice* inputs, BatchDevice* outputs, BatchDevice* mask, float p);

// Dense
__global__ void DenseKernelF(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* weights, BatchDevice* bias);
__global__ void DenseKernelB(BatchDevice* inputs, BatchDevice* outputs, BatchDevice* weights, BatchDevice* bias, float L, float R2);
__global__ void weightUpdate(BatchDevice* weights, float lr);
__global__ void biasUpdate(BatchDevice* bias, float lr);

// Output - softmax + CCE
__global__ void OutputKernel(BatchDevice* outputs, BatchDevice* inputs, BatchDevice* oneHot);
__global__ void CopyGrad(BatchDevice* input, BatchDevice* output);

// Evaluation Matrix 
__global__ void EvaluateKernel(BatchDevice* predicted, BatchDevice* actual, BatchDevice* matrix);

#endif 

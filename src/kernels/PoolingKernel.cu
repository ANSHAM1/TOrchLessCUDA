#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <limits>




__global__ void maxPoolingKernel(const float* input, float* output, size_t N, size_t C, size_t H, size_t W, size_t OH, size_t OW,
    size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = N * C * OH * OW;

    if (idx >= total)
        return;


    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;

    size_t c = (idx / (OW * OH)) % C;
    size_t n = idx / (OW * OH * C);

    float maxValue = -FLT_MAX;


    for (size_t kh = 0; kh < KH; kh++)
        for (size_t kw = 0; kw < KW; kw++) {
            int ih = oh * SH + kh - PH;
            int iw = ow * SW + kw - PW;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                size_t inputIndex = ((n * C + c) * H + ih) * W + iw;

                maxValue = fmaxf(maxValue, input[inputIndex]);
            }
        }
    
    output[idx] = maxValue;
}


__global__ void avgPoolingKernel(const float* input, float* output, size_t N, size_t C, size_t H, size_t W, size_t OH, 
    size_t OW, size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = N * C * OH * OW;

    if (idx >= total)
        return;


    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;

    size_t c = (idx / (OW * OH)) % C;
    size_t n = idx / (OW * OH * C);


    float sum = 0.0f;

    size_t count = 0;

    for (size_t kh = 0; kh < KH; kh++)
        for (size_t kw = 0; kw < KW; kw++) {
            int ih = oh * SH + kh - PH;
            int iw = ow * SW + kw - PW;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                size_t inputIndex = ((n * C + c) * H + ih) * W + iw;

                sum += input[inputIndex];
                count++;
            }
        }
    
    output[idx] = sum / count;
}


void poolingForward(const Tensor& input, Tensor& output, size_t kernelHeight, size_t kernelWidth, size_t strideHeight,
    size_t strideWidth, size_t paddingHeight, size_t paddingWidth, const std::string& type) {

    const auto& shape = input.shape();

    size_t N = shape[0];
    size_t C = shape[1];
    size_t H = shape[2];
    size_t W = shape[3];


    const auto& outShape = output.shape();

    size_t OH = outShape[2];
    size_t OW = outShape[3];

    size_t total = N * C * OH * OW;


    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (type == "max")
        ExecuteKernel("maxPoolingKernel", blocks, threads, 0, 0, maxPoolingKernel, input.data(), output.data(), N, C, H, W,
            OH, OW, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingHeight, paddingWidth);

    else if (type == "avg")
        ExecuteKernel("avgPoolingKernel", blocks, threads, 0, 0, avgPoolingKernel, input.data(), output.data(), N, C, H, W,
            OH, OW, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingHeight, paddingWidth);
    
    else
        throw std::runtime_error("Unsupported pooling type");
}
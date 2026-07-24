#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <limits>
#include <cfloat>
#include <float.h>




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




__global__ void maxPoolingBackwardKernel(const float* input, const float* gradOutput, float* gradInput, size_t Batch, size_t Channels,
    size_t InputHeight, size_t InputWidth, size_t OutputHeight, size_t OutputWidth, size_t KernelHeight, size_t KernelWidth,
    size_t StrideHeight, size_t StrideWidth, size_t PaddingHeight, size_t PaddingWidth) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = Batch * Channels * InputHeight * InputWidth;
    if (idx >= total)
        return;

    size_t w = idx % InputWidth;
    size_t h = (idx / InputWidth) % InputHeight;
    size_t c = (idx / (InputHeight * InputWidth)) % Channels;
    size_t n = idx / (Channels * InputHeight * InputWidth);


    float gradient = 0.0f;

    for (size_t oh = 0; oh < OutputHeight; oh++) {
        for (size_t ow = 0; ow < OutputWidth; ow++) {
            int startH = oh * StrideHeight - PaddingHeight;
            int startW = ow * StrideWidth - PaddingWidth;

            if (h < startH || h >= startH + KernelHeight || w < startW || w >= startW + KernelWidth)
                continue;


            float maxValue = -FLT_MAX;

            int maxH = -1;
            int maxW = -1;


            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    int ih = startH + kh;
                    int iw = startW + kw;

                    if (ih < 0 || iw < 0 || ih >= InputHeight || iw >= InputWidth)
                        continue;

                    size_t inputIndex = ((n * Channels + c) * InputHeight + ih) * InputWidth + iw;

                    float value = input[inputIndex];
                    if (value > maxValue) {
                        maxValue = value;
                        maxH = ih;
                        maxW = iw;
                    }
                }
            }

            // This input pixel was max
            if (maxH == h && maxW == w) {
                size_t outputIndex = ((n * Channels + c) * OutputHeight + oh) * OutputWidth + ow;

                gradient += gradOutput[outputIndex];
            }
        }
    }

    gradInput[idx] = gradient;
}


__global__ void avgPoolingBackwardKernel(const float* gradOutput, float* gradInput, size_t Batch, size_t Channels, 
    size_t InputHeight, size_t InputWidth, size_t OutputHeight, size_t OutputWidth, size_t KernelHeight, size_t KernelWidth,
    size_t StrideHeight, size_t StrideWidth, size_t PaddingHeight, size_t PaddingWidth) {
    
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = Batch * Channels * OutputHeight * OutputWidth;
    if (idx >= total)
        return;

    size_t w = idx % InputWidth;
    size_t h = (idx / InputWidth) % InputHeight;
    size_t c = (idx / (InputHeight * InputWidth)) % Channels;
    size_t n = idx / (Channels * InputHeight * InputWidth);


    float gradient = 0.0f;

    float scale = 1.0f / (KernelHeight * KernelWidth);

    for (size_t oh = 0; oh < OutputHeight; oh++) {
        for (size_t ow = 0; ow < OutputWidth; ow++) {
            int startH = oh * StrideHeight - PaddingHeight;
            int startW = ow * StrideWidth -  PaddingWidth;

            if (h < startH || h >= startH + KernelHeight || w < startW || w >= startW + KernelWidth)
                continue;

            size_t outputIndex = ((n * Channels + c) * OutputHeight + oh) * OutputWidth + ow;

            gradient += gradOutput[outputIndex] * scale;
        }
    }

    gradInput[idx] = gradient;
}


void poolingBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput, size_t KernelHeight,
    size_t KernelWidth, size_t StrideHeight, size_t StrideWidth, size_t PaddingHeight, size_t PaddingWidth, const std::string& type) {

    size_t Batch = input.shape()[0];
    size_t Channels = input.shape()[1];

    size_t InputHeight = input.shape()[2];
    size_t InputWidth = input.shape()[3];

    size_t OutputHeight = output.shape()[2];
    size_t OutputWidth = output.shape()[3];

    size_t total = Batch * Channels * InputHeight * InputWidth;


    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (type == "max")
        ExecuteKernel("maxPoolingBackwardKernel", blocks, threads, 0, 0, maxPoolingBackwardKernel, input.data(), 
            gradOutput.data(), gradInput.data(), Batch, Channels, InputHeight, InputWidth, OutputHeight, 
            OutputWidth, KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth);

    else if (type == "avg")
        ExecuteKernel("avgPoolingBackwardKernel", blocks, threads, 0, 0, avgPoolingBackwardKernel, gradOutput.data(), 
            gradInput.data(), Batch, Channels, InputHeight, InputWidth, OutputHeight, OutputWidth, KernelHeight, 
            KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth);

    else
        throw std::runtime_error("Unsupported pooling backward");
}
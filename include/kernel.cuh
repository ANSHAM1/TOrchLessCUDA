#pragma once

#include <cstddef>
#include <string>


// util kernels

void tensorAssign(float* ptr, std::size_t value, std::size_t n);

void tensorAssignRandom(float* data, std::size_t size, unsigned long long seed);

void uniformDist(float* data, size_t size, unsigned long long seed, float min, float max);


class Tensor;
// kernels



// Convolution kernels
void conv2dForward(const Tensor& input, const Tensor& kernel, const Tensor& bias, Tensor& output, 
    size_t stride, size_t padding);

void conv2dInputBackward(const Tensor& gradOutput, const Tensor& kernel, Tensor& gradInput, size_t stride, size_t padding);
void conv2dWeightBackward(const Tensor& input, const Tensor& gradOutput, Tensor& gradKernel, size_t stride, size_t padding);
void conv2dBiasBackward(const Tensor& gradOutput, Tensor& gradBias);



// Activation kernels
void activationForward(const Tensor& input, Tensor& output, const std::string& type);

void activationBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, 
    Tensor& gradInput, const std::string& type);



// Pooling kernels
void poolingForward(const Tensor& input, Tensor& output, size_t kernelHeight, size_t kernelWidth, size_t strideHeight, 
    size_t strideWidth, size_t paddingHeight, size_t paddingWidth, const std::string& type);

void poolingBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput, size_t KernelHeight, 
    size_t KernelWidth, size_t StrideHeight, size_t StrideWidth, size_t PaddingHeight, size_t PaddingWidth, const std::string& Type);



// dense kernels
void denseForward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

void denseInputBackward(const Tensor& gradOutput, const Tensor& weight, Tensor& gradInput);
void denseWeightBackward(const Tensor& input, const Tensor& gradOutput, Tensor& gradWeight);
void denseBiasBackward( const Tensor& gradOutput, Tensor& gradBias);



void outputForward(const Tensor& input, Tensor& output, const std::string& type);




// losses

float SoftmaxCCELoss(const Tensor& logits, const Tensor& target);

void SoftmaxCCELossBackward(const Tensor& prediction, const Tensor& target, Tensor& grad);

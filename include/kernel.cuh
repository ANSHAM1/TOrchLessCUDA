#pragma once

#include <cstddef>
#include <string>


// util kernels

void tensorAssign(float* ptr, std::size_t value, std::size_t n);

void tensorAssignRandom(float* data, std::size_t size, unsigned long long seed);

void uniformDist(float* data, size_t size, unsigned long long seed, float min, float max);




class Tensor;

// kernels

void conv2dForward(const Tensor& input, const Tensor& kernel, const Tensor& bias, Tensor& output, 
    size_t stride, size_t padding);

void activationForward(const Tensor& input, Tensor& output, const std::string& type);

void poolingForward(const Tensor& input, Tensor& output, size_t kernelHeight, size_t kernelWidth, size_t strideHeight, 
    size_t strideWidth, size_t paddingHeight, size_t paddingWidth, const std::string& type);

void denseForward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

void outputForward(const Tensor& input, Tensor& output, const std::string& type);
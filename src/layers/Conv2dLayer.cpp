#include "module.hpp"

#include <random>




Conv2dLayer::Conv2dLayer(const std::vector<size_t>& inputShape, const std::vector<size_t>& kernelShape,
    size_t stride, size_t padding) : Kernel(kernelShape), Bias({kernelShape[0]}), Stride(stride), Padding(padding) {

    if (inputShape.size() != 4)
        throw std::invalid_argument("Conv2D input must be NCHW");

    if (kernelShape.size() != 4)
        throw std::invalid_argument("Conv2D kernel must be OIHW");

    KernelGrad.allocate(kernelShape);
    BiasGrad.allocate({ kernelShape[0] });

    Bias.fill(0.0f);


    size_t N = inputShape[0];
    size_t C = inputShape[1];
    size_t H = inputShape[2];
    size_t W = inputShape[3];

    size_t K = kernelShape[0];
    size_t KH = kernelShape[2];
    size_t KW = kernelShape[3];

    size_t OH = (H + 2 * Padding - KH) / Stride + 1;
    size_t OW = (W + 2 * Padding - KW) / Stride + 1;
    OutputShape = { N, K, OH, OW };

    float fanIn = static_cast<float>(C * KH * KW);
    float limit = sqrtf(6.0f / fanIn);

    uniformDist(Kernel.data(), Kernel.numel(), 1234ULL, -limit, limit);
}


void Conv2dLayer::forward(const Tensor& input, Tensor& output) {
    conv2dForward(input, Kernel, Bias, output, Stride, Padding);
}


void Conv2dLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    gradInput.reshape(input.shape());

    conv2dInputBackward(gradOutput, Kernel, gradInput, Stride, Padding);

    conv2dWeightBackward(input, gradOutput, KernelGrad, Stride, Padding);

    conv2dBiasBackward(gradOutput, BiasGrad);
}
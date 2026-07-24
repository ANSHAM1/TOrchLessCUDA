#include "layer.hpp"

#include <random>



Conv2dLayer::Conv2dLayer(const std::vector<size_t>& InputShape, const std::vector<size_t>& KernelShape,
    size_t stride, size_t padding) : Kernel(KernelShape), Bias({KernelShape[0]}), Stride(stride), Padding(padding) {

    if (InputShape.size() != 4)
        throw std::invalid_argument("Conv2D input must be NCHW");

    if (KernelShape.size() != 4)
        throw std::invalid_argument("Conv2D kernel must be OIHW");


    size_t N = InputShape[0];
    size_t C = InputShape[1];
    size_t H = InputShape[2];
    size_t W = InputShape[3];

    size_t K = KernelShape[0];
    size_t KH = KernelShape[2];
    size_t KW = KernelShape[3];

    size_t OH = (H + 2 * Padding - KH) / Stride + 1;
    size_t OW = (W + 2 * Padding - KW) / Stride + 1;
    OutputShape = { N, K, OH, OW };

    Bias.fill(0.0f);

    float fanIn = static_cast<float>(C * KH * KW);
    float limit = sqrtf(6.0f / fanIn);

    uniformDist(Kernel.data(), Kernel.numel(), 1234ULL, -limit, limit);
}



void Conv2dLayer::forward(const Tensor& input, Tensor& output) {
    conv2dForward(input, Kernel, Bias, output, Stride, Padding);
}
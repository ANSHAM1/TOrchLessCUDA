#include "module.hpp"




PoolingLayer::PoolingLayer(const std::vector<size_t>& inputShape, size_t kh, size_t kw, size_t sh, size_t sw, 
    size_t ph, size_t pw, const std::string& type ) 
    : KernelHeight(kh), KernelWidth(kw), StrideHeight(sh), StrideWidth(sw), PaddingHeight(ph), PaddingWidth(pw), Type(type) {

    if (Type != "max" && Type != "avg")
        throw std::runtime_error("Unsupported pooling type");


    size_t H = (inputShape[2] + 2 * PaddingHeight - KernelHeight) / StrideHeight + 1;
    size_t W = (inputShape[3] + 2 * PaddingWidth - KernelWidth) / StrideWidth + 1;

    OutputShape = { inputShape[0], inputShape[1], H, W };
}


void PoolingLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    poolingForward(input, output, KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth, Type);
}


void PoolingLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    gradInput.reshape(input.shape());

    poolingBackward(input, output, gradOutput, gradInput, KernelHeight, KernelWidth, StrideHeight, 
        StrideWidth, PaddingHeight, PaddingWidth, Type);
}
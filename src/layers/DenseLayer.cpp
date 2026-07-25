#include "module.hpp"

#include <cmath>




DenseLayer::DenseLayer(const std::vector<size_t>& inputShape, size_t outNumFeature) : OutFeatures(outNumFeature) {
    if (inputShape.size() != 2)
        throw std::runtime_error("DenseLayer expects flattened input [Batch, Features]");


    size_t InFeatures = inputShape[1];

    Weight.allocate({ InFeatures, OutFeatures });
    Bias.allocate({ OutFeatures });

    float limit = sqrtf(6.0f / (InFeatures + OutFeatures));
    uniformDist(Weight.data(), Weight.numel(), 1234ULL, -limit, limit);

    Bias.fill(0.0f);

    OutputShape = { inputShape[0], OutFeatures };
}


void DenseLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    denseForward(input, Weight, Bias, output);
}


void DenseLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {  
    gradInput.reshape(input.shape());

    denseInputBackward(gradOutput, Weight, gradInput);

    denseWeightBackward(input, gradOutput, WeightGrad);

    denseBiasBackward(gradOutput, BiasGrad);
}
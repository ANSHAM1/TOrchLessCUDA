#include "layer.hpp"




DenseLayer::DenseLayer(const std::vector<size_t>& InputShape, size_t OutNumFeature) : OutFeatures(OutNumFeature) {
    if (InputShape.size() != 2)
        throw std::runtime_error("DenseLayer expects flattened input [Batch, Features]");


    size_t InFeatures = InputShape[1];

    Weight.allocate({ InFeatures, OutFeatures });
    Bias.allocate({ OutFeatures });

    float limit = sqrtf(6.0f / (InFeatures + OutFeatures));
    uniformDist(Weight.data(), Weight.numel(), 1234ULL, -limit, limit);

    Weight.debug_print("Dense Weight", 10);

    Bias.fill(0.0f);

    OutputShape = { InputShape[0], OutFeatures };
}


void DenseLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    denseForward(input, Weight, Bias, output);
}
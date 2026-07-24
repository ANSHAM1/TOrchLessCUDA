#include "layer.hpp"



FlattenLayer::FlattenLayer(const std::vector<size_t>& inputShape) {
    if (inputShape.size() < 2)
        throw std::invalid_argument("Flatten requires input dimension >= 2");

    size_t Batch = inputShape[0];

    size_t Features = 1;
    for (size_t i = 1; i < inputShape.size(); i++)
        Features *= inputShape[i];

    OutputShape = { Batch, Features };
}



void FlattenLayer::forward(const Tensor& input, Tensor& output) {
    output = input.view(OutputShape);
}


void FlattenLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    gradInput = gradOutput.view(input.shape());
}
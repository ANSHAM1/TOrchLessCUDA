#include "layer.hpp"



ActivationLayer::ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type) : Type(type) {
    OutputShape = InputShape;
}


void ActivationLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    activationForward(input, output, Type);
}


void ActivationLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    gradInput.reshape(input.shape());

    activationBackward(input, output, gradOutput, gradInput, Type);
}
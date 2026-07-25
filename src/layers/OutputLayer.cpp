#include "module.hpp"




OutputLayer::OutputLayer(const std::vector<size_t>& inputShape, const std::string& type) : Type(type) {
    if (Type != "Softmax" && Type != "Sigmoid")
        throw std::runtime_error("Unsupported output layer type.");
    
    OutputShape = inputShape;
}


void OutputLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    outputForward(input, output, Type);
}


void OutputLayer::backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    gradInput = gradOutput.view(gradOutput.shape());
}
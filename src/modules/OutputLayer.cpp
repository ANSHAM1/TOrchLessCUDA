#include "layer.hpp"



OutputLayer::OutputLayer(const std::vector<size_t>& InputShape, const std::string& type) : Type(type) {
    if (Type != "Softmax" && Type != "Sigmoid" && Type != "Identity")
        throw std::runtime_error("Unsupported output layer type.");
    
    OutputShape = InputShape;
}


void OutputLayer::forward(const Tensor& input, Tensor& output) {
    output.reshape(OutputShape);

    outputForward(input, output, Type);
}
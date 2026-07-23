#include "layer.hpp"



ActivationLayer::ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type) : Type(type) {
    OutputShape = InputShape;
}


void ActivationLayer::forward(const Tensor& input, Tensor& output, ExecutionContext& context) {
    output.reshape(OutputShape);

    if (Type == "relu")
    {
        /* reluForward(
            input,
            output,
            context
        );*/
    }
    else if (Type == "sigmoid")
    {
        /* sigmoidForward(
            input,
            output,
            context
        );*/
    }
    else if (Type == "tanh")
    {
        /* tanhForward(
            input,
            output,
            context
        );*/
    }
    else
    {
        throw std::runtime_error(
            "Unsupported activation type: " + Type
        );
    }
}
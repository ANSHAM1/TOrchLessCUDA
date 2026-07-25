#include "module.hpp"

#include <iostream>




void ExecutionContext::compileInference(const std::vector<std::unique_ptr<Layer>>& Layers, const std::vector<size_t>& InputShape) {
    if (IsCompiled)
        return;

    // Find Largest Activation Tensor
    std::vector<size_t> LargestShape = InputShape;

    size_t MaxElementCount = Tensor::shape_product(InputShape);
    for (const auto& Layer : Layers) {
        const auto& Shape = Layer->getOutputShape();

        size_t Elements = Tensor::shape_product(Shape);
        if (Elements > MaxElementCount) {
            MaxElementCount = Elements;
            LargestShape = Shape;
        }
    }

    // Allocate Ping Pong Buffers
    WorkspaceA.allocate(LargestShape);
    WorkspaceB.allocate(LargestShape);

    IsCompiled = true;
}




void ExecutionContext::compileTraining(const std::vector<std::unique_ptr<Layer>>& Layers, const std::vector<size_t>& InputShape) {
    if (IsCompiled)
        return;

    Activations.clear();
    Gradients.clear();

    std::vector<size_t> Shape = InputShape;

    // Allocate Activation Buffers
    Activations.emplace_back();
    Activations.back().allocate(Shape);


    for (const auto& layer : Layers) {
        Shape = layer->getOutputShape();

        Activations.emplace_back();

        if (!layer->isViewOperation())
            Activations.back().allocate(Shape);
    }


    // Allocate Gradient Buffers
    Gradients.reserve(Activations.size());

    Gradients.emplace_back();
    Gradients.back().allocate(InputShape);


    for (const auto& layer : Layers) {
        Gradients.emplace_back();

        Gradients.back().allocate(layer->getOutputShape());
    }


    IsCompiled = true;
}
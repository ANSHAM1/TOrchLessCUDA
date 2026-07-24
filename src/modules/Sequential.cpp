#include "layer.hpp"



void Sequential::Input(const std::vector<size_t>& Shape, size_t Batch) {
    if (Shape.size() != 4)
        throw std::invalid_argument("Input shape must be (N, C, H, W)");

    if (Batch == 0)
        throw std::invalid_argument("Batch size cannot be zero.");


    InputShape = { Batch, Shape[1], Shape[2], Shape[3] };

    BatchSize = Batch;

    NumBatch = Shape[0] / Batch;
}



// Compile Inference
void Sequential::compileInference(ExecutionContext& Context) {
    if (Context.IsCompiled)
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
    Context.WorkspaceA.allocate(LargestShape);
    Context.WorkspaceB.allocate(LargestShape);

    Context.IsCompiled = true;
}


// Compile Training
void Sequential::compileTraining(ExecutionContext& Context) {
    if (Context.IsCompiled)
        return;

    Context.Activations.clear();
    Context.Gradients.clear();

    std::vector<size_t> Shape = InputShape;

    // Allocate Activation Buffers
    Context.Activations.emplace_back();
    Context.Activations.back().allocate(Shape);

    for (const auto& layer : Layers) {
        Shape = layer->getOutputShape();

        Context.Activations.emplace_back();

        if (!layer->isViewOperation())
            Context.Activations.back().allocate(Shape);
    }


    // Allocate Gradient Buffers
    Context.Gradients.reserve(Context.Activations.size());

    Context.Gradients.emplace_back();
    Context.Gradients.back().allocate(InputShape);

    for (const auto& layer : Layers) {

        Context.Gradients.emplace_back();

        if (!layer->isViewOperation())
            Context.Gradients.back().allocate(layer->getOutputShape());
    }


    // Shared Scratch Workspace
    std::vector<size_t> LargestShape = InputShape;
    size_t MaxElements = Tensor::shape_product(InputShape);

    for (const auto& Activation : Context.Activations) {
        size_t Elements = Activation.numel();

        if (Elements > MaxElements) {
            MaxElements = Elements;
            LargestShape = Activation.shape();
        }
    }

    Context.Workspace.allocate(LargestShape);

    Context.IsCompiled = true;
}
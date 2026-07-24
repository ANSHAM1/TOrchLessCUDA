#include "layer.hpp"




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


Tensor& Sequential::inference(ExecutionContext& Context, const Tensor& Input) {
    Tensor* Current = const_cast<Tensor*>(&Input);

    Tensor* Next = &Context.WorkspaceA;

    bool useA = true;
    for (auto& Layer : Layers) {
        if (Layer->isViewOperation())
        {
            Tensor View;

            Layer->forward(
                *Current,
                View
            );

            Current = new Tensor(std::move(View));

            continue;
        }

        /*
            Output tensor gets only reshaped.
            Memory is already allocated in compile().
        */

        Next->reshape(Layer->getOutputShape());

        Layer->forward(*Current, *Next);


        Current = Next;


        if (useA)
            Next = &Context.WorkspaceB;
        else
            Next = &Context.WorkspaceA;

        useA = !useA;
    }

    return *Current;
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


// -------------------------------------------------------------------------------------------------------------


void Sequential::Input(const std::vector<size_t>& Shape, size_t Batch) {
    if (Shape.size() != 4)
        throw std::invalid_argument("Input shape must be (N, C, H, W)");

    if (Batch == 0)
        throw std::invalid_argument("Batch size cannot be zero.");


    InputShape = { Batch, Shape[1], Shape[2], Shape[3] };

    BatchSize = Batch;

    NumBatch = Shape[0] / Batch;

    CurrentShape = InputShape;
}


void Sequential::Conv2D(const std::vector<size_t>& KernelShape, size_t Stride, size_t Padding) {
    if (KernelShape.size() != 4)
        throw std::invalid_argument("Kernel shape must be OIHW");

    auto Layer = std::make_unique<Conv2dLayer>(CurrentShape, KernelShape, Stride, Padding);

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::ReLU() {
    auto Layer = std::make_unique<ActivationLayer>(CurrentShape, "relu");

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::Sigmoid() {
    auto Layer = std::make_unique<ActivationLayer>(CurrentShape, "sigmoid");

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::Tanh() {
    auto Layer = std::make_unique<ActivationLayer>(CurrentShape, "tanh");

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::MaxPooling(size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW) {
    auto Layer = std::make_unique<PoolingLayer>(CurrentShape, KH, KW, SH, SW, PH, PW, "max");

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::AvgPooling(size_t KH, size_t KW, size_t SH, size_t SW, size_t PH, size_t PW) {
    auto Layer = std::make_unique<PoolingLayer>(CurrentShape, KH, KW, SH, SW, PH, PW, "avg");

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


void Sequential::Flatten() {
    auto Layer = std::make_unique<FlattenLayer>(CurrentShape);

    CurrentShape = Layer->getOutputShape();


    Layers.emplace_back(std::move(Layer));
}


void Sequential::Dense(size_t OutFeatures) {
    auto Layer = std::make_unique<DenseLayer>(CurrentShape, OutFeatures);

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back( std::move(Layer));
}


void Sequential::Output(const std::string& Type) {
    auto Layer = std::make_unique<OutputLayer>(CurrentShape, Type);

    CurrentShape = Layer->getOutputShape();

    Layers.emplace_back(std::move(Layer));
}


//------------------------------------------------------------------------------------------------------------ -


Tensor& Sequential::Predict(ExecutionContext& Context, const Tensor& Input) {
    compileInference(Context);

    return inference(Context, Input);
}
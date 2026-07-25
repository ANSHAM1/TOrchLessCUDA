#include "module.hpp"
#include "optimizer.hpp"

#include <iostream>




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




//-------------------------------------------------------------------------------------------------------------




Tensor& Sequential::Predict(ExecutionContext& Context, const Tensor& Input) {
    Context.compileInference(Layers, InputShape);

    return forwardInference(Context, Input);
}




Tensor& Sequential::forwardInference(ExecutionContext& Context, const Tensor& Input) {
    Tensor* Current = const_cast<Tensor*>(&Input);

    Tensor* Next = &Context.WorkspaceA;

    bool useA = true;
    for (auto& Layer : Layers) {
        if (Layer->isViewOperation()) {
            Tensor View;

            Layer->forward(*Current, View);

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




//-------------------------------------------------------------------------------------------------------------




float Sequential::Train(ExecutionContext& Context, Loss& LossFun, Optimizer& Optimizer, const Tensor& Input, const Tensor& Label) {
    Context.compileTraining(Layers, InputShape);

    Tensor& Prediction = forwardpropagation(Context, Input);


    float LossValue = LossFun.forward(Prediction, Label);
    Tensor LossGradient(Prediction.shape());
    LossFun.backward(Prediction, Label, LossGradient);


    backpropagation(Context, Input, LossGradient);


    Optimizer.step(parameters(), gradients());

    return LossValue;
}




Tensor& Sequential::forwardpropagation(ExecutionContext& Context, const Tensor& Input) {
    Context.Activations[0] = Input.view(Input.shape());
    Tensor* Current = &Context.Activations[0];

    for (size_t i = 0; i < Layers.size(); i++) {
        Tensor& Next = Context.Activations[i + 1];

        if (Layers[i]->isViewOperation())
            Layers[i]->forward(*Current, Next );

        else {
            Next.reshape(Layers[i]->getOutputShape());

            Layers[i]->forward(*Current, Next);
        }

        Current = &Next;
    }

    return *Current;
}




void Sequential::backpropagation(ExecutionContext& Context, const Tensor& Input, Tensor& GradOutput) {
    Tensor* CurrentGradient = &GradOutput;

    for (int i = static_cast<int>(Layers.size()) - 1; i >= 0; i--) {
        Tensor& InputActivation = Context.Activations[i];
        Tensor& OutputActivation = Context.Activations[static_cast<std::vector<Tensor, std::allocator<Tensor>>::size_type>(i) + 1];

        Tensor& NextGradient = Context.Gradients[i];


        NextGradient.reshape(InputActivation.shape());

        Layers[i]->backward(InputActivation, OutputActivation, *CurrentGradient, NextGradient);

        CurrentGradient = &NextGradient;
    }
}
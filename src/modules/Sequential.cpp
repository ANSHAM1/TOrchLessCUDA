#include "module.hpp"

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




void Sequential::Train(ExecutionContext& Context, const Tensor& Input, const Tensor& Label) {
    Context.compileTraining(Layers, InputShape);


    Tensor& Prediction = forwardTraining(Context, Input);


    Prediction.debug_print("Prediction", 10);


    Loss loss;


    float LossValue = loss.forward(Prediction, Label);


    std::cout << "Loss: "
        << LossValue
        << std::endl;



    Tensor LossGradient(Prediction.shape());


    loss.backward(
        Prediction,
        Label,
        LossGradient
    );


    LossGradient.debug_print("Loss Gradient", 10);



    backpropagation(
        Context,
        Input,
        LossGradient
    );
}


//-------------------------------------------------------------------------------------------------------------


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




Tensor& Sequential::forwardTraining(ExecutionContext& Context, const Tensor& Input) {

    Context.Activations[0] = Input.view(Input.shape());


    Tensor* Current = &Context.Activations[0];


    for (size_t i = 0; i < Layers.size(); i++)
    {

        Tensor& Next = Context.Activations[i + 1];


        if (Layers[i]->isViewOperation())
        {
            Layers[i]->forward(
                *Current,
                Next
            );
        }
        else
        {
            Next.reshape(Layers[i]->getOutputShape());

            Layers[i]->forward(
                *Current,
                Next
            );
        }


        Current = &Next;


        if (i < 3)
        {
            std::cout << "Forward layer: "
                << i
                << std::endl;

            Next.debug_print("Activation", 10);
        }
    }


    return *Current;
}


void Sequential::backpropagation(ExecutionContext& Context, const Tensor& Input, Tensor& GradOutput)
{
    Tensor* CurrentGradient = &GradOutput;


    for (int i = static_cast<int>(Layers.size()) - 1; i >= 0; i--)
    {

        Tensor& InputActivation = Context.Activations[i];

        Tensor& OutputActivation = Context.Activations[i + 1];


        Tensor& NextGradient = Context.Gradients[i];


        NextGradient.reshape(
            InputActivation.shape()
        );


        std::cout << "Layer " << i << " input shape: ";
        for (auto x : InputActivation.shape())
            std::cout << x << " ";
        std::cout << std::endl;


        Layers[i]->backward(
            InputActivation,
            OutputActivation,
            *CurrentGradient,
            NextGradient
        );


        CurrentGradient = &NextGradient;


        std::cout << "Backward layer: "
            << i
            << std::endl;


        NextGradient.debug_print("Gradient", 10);
    }
}


// -------------------------------------------------------------------------------------------------------------
#pragma once

#include "tensor.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>




class Optimizer;




class Layer;




class ExecutionContext {

public:

    Tensor WorkspaceA;
    Tensor WorkspaceB;

    std::vector<Tensor> Activations;
    std::vector<Tensor> Gradients;

    bool IsCompiled = false;

public:

    void compileInference(
        const std::vector<std::unique_ptr<Layer>>& Layers,
        const std::vector<size_t>& InputShape
    );


    void compileTraining(
        const std::vector<std::unique_ptr<Layer>>& Layers,
        const std::vector<size_t>& InputShape
    );

};



 
class Loss {

private:

    std::string Type;

public:

    Loss(const std::string& type = "CCE") : Type(type) {}


    float forward(const Tensor& prediction, const Tensor& target) const;

    void backward(const Tensor& prediction, const Tensor& target, Tensor& grad) const;

};




class Layer {

protected:

    std::vector<size_t> OutputShape;

public:

    virtual ~Layer() = default;

    virtual void initialize() {}

    virtual void forward(const Tensor& input, Tensor& output) = 0;

    virtual void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {}

    virtual std::vector<Tensor*> parameters() {
        return {};
    }

    virtual std::vector<Tensor*> gradients() {
        return {};
    }

    virtual bool isViewOperation() const {
        return false;
    }

    const std::vector<size_t>& getOutputShape() const {
        return OutputShape;
    }

};




class Conv2dLayer : public Layer {

public:

    Tensor Kernel;
    Tensor Bias;

    Tensor KernelGrad;
    Tensor BiasGrad;

    size_t Stride;
    size_t Padding;

    Conv2dLayer(const std::vector<size_t>& inputShape, const std::vector<size_t>& kernelShape, 
        size_t stride, size_t padding);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

    std::vector<Tensor*> parameters() override {
        return { &Kernel, &Bias };
    }

    std::vector<Tensor*> gradients() override {
        return { &KernelGrad, &BiasGrad };
    }

};




class ActivationLayer : public Layer {

public:

    std::string Type;

    ActivationLayer(const std::vector<size_t>& inputShape, const std::string& type);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

};




class PoolingLayer : public Layer {

public:

    size_t KernelHeight;
    size_t KernelWidth;

    size_t StrideHeight;
    size_t StrideWidth;

    size_t PaddingHeight;
    size_t PaddingWidth;

    std::string Type;

    PoolingLayer(const std::vector<size_t>& inputShape, size_t kh, size_t kw, size_t sh,size_t sw, 
        size_t ph, size_t pw, const std::string& type);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

};




class FlattenLayer : public Layer {

public:

    FlattenLayer(const std::vector<size_t>& inputShape);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

    bool isViewOperation() const override {
        return true;
    }

};




class DenseLayer : public Layer {

public:

    Tensor Weight;
    Tensor Bias;

    Tensor WeightGrad;
    Tensor BiasGrad;

    size_t OutFeatures;

    DenseLayer(const std::vector<size_t>& inputShape, size_t outNumFeature);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

    std::vector<Tensor*> parameters() override {
        return { &Weight, &Bias };
    }

    std::vector<Tensor*> gradients() override {
        return { &WeightGrad, &BiasGrad };
    }

};




class OutputLayer : public Layer {

public:

    std::string Type;

    OutputLayer(const std::vector<size_t>& inputShape, const std::string& type = "Softmax");

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

};




class Sequential {

private:

    std::vector<std::unique_ptr<Layer>> Layers;

    std::vector<size_t> InputShape;
    std::vector<size_t> CurrentShape;

    size_t BatchSize = 1;
    size_t NumBatch = 0;

private:

    Tensor& forwardInference(ExecutionContext& Context, const Tensor& Input);

    Tensor& forwardTraining(ExecutionContext& Context, const Tensor& Input);
    void backpropagation(ExecutionContext& Context, const Tensor& Input, Tensor& GradOutput);

public:

    Sequential() = default;


    void Input(const std::vector<size_t>& Shape, size_t Batch);

    void Conv2D(const std::vector<size_t>& KernelShape, size_t Stride = 1, size_t Padding = 0);

    void ReLU();
    void Sigmoid();
    void Tanh();

    void MaxPooling(size_t KH, size_t KW, size_t SH = 1, size_t SW = 1, size_t PH = 0, size_t PW = 0);
    void AvgPooling(size_t KH, size_t KW, size_t SH = 1, size_t SW = 1, size_t PH = 0, size_t PW = 0);

    void Flatten();

    void Dense(size_t OutFeatures);

    void Output(const std::string& type = "Softmax");

    


    Tensor& Predict(ExecutionContext& Context, const Tensor& Input);

    void Train(ExecutionContext& Context, Optimizer& Otm, const Tensor& Input, const Tensor& Label);

    


    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> Params;

        for (auto& Layer : Layers) {
            auto P = Layer->parameters();

            Params.insert(Params.end(), P.begin(), P.end());
        }

        return Params;
    }


    std::vector<Tensor*> gradients() {
        std::vector<Tensor*> Grads;

        for (auto& Layer : Layers) {
            auto G = Layer->gradients();

            Grads.insert(Grads.end(), G.begin(), G.end());
        }

        return Grads;
    }

};
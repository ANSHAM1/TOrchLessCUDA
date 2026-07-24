#pragma once

#include "tensor.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>






// Execution Context
class ExecutionContext {
public:

    // Inference Workspace
    Tensor WorkspaceA;
    Tensor WorkspaceB;

    // Training Buffers
    std::vector<Tensor> Activations;
    std::vector<Tensor> Gradients;

    // Shared Scratch Memory
    Tensor Workspace;


    bool IsCompiled = false;
};






// Loss 
class Loss {
private:

    std::string Type;

public:

    Loss(const std::string& type = "CCE") : Type(type) {}

    // Forward loss value
    float forward(const Tensor& prediction, const Tensor& target) const;

    // Gradient w.r.t prediction
    void backward(const Tensor& prediction, const Tensor& target, Tensor& grad) const;
};







// Base Layer
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




// Convolution Layer
class Conv2dLayer : public Layer {
public:

    //Parameters:
    Tensor Kernel;
    Tensor Bias;

    //Gradients:
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


// Activation Layer
class ActivationLayer : public Layer {
public:

    std::string Type;

    ActivationLayer(const std::vector<size_t>& inputShape, const std::string& type);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;
};


// Pooling Layer
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


// Flatten Layer
class FlattenLayer : public Layer {
public:

    FlattenLayer(const std::vector<size_t>& inputShape);

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;

    bool isViewOperation() const override {
        return true;
    }
};


// Dense / Linear Layer
class DenseLayer : public Layer {
public:

    //Parameters:
    Tensor Weight;
    Tensor Bias;

    //Gradients:
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


// Output Layer
class OutputLayer : public Layer {
public:

    std::string Type;

    OutputLayer(const std::vector<size_t>& inputShape, const std::string& type = "Softmax");

    void forward(const Tensor& input, Tensor& output) override;

    void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) override;
};





// Sequential Model
class Sequential {
private:

    std::vector<std::unique_ptr<Layer>> Layers;

    std::vector<size_t> InputShape;
    std::vector<size_t> CurrentShape;

    size_t BatchSize = 1;
    size_t NumBatch = 0;

private:

    // Inference
    void compileInference(ExecutionContext& Context);
    Tensor& forwardInference(ExecutionContext& Context, const Tensor& Input);

    // Training
    void compileTraining(ExecutionContext& Context);
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

    void Train(ExecutionContext& Context, const Tensor& Input, const Tensor& Label);

    


    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> Params;

        for (auto& Layer : Layers) {
            auto P = Layer->parameters();

            Params.insert(Params.end(), P.begin(), P.end());
        }

        return Params;
    }
};
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



// Base Layer
class Layer {
protected:

    std::vector<size_t> OutputShape;

public:

    virtual ~Layer() = default;

    virtual void initialize() {}


    virtual void forward(const Tensor& input, Tensor& output, ExecutionContext& context) = 0;

    virtual void backward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, 
        Tensor& gradInput, ExecutionContext& context) {}

    virtual std::vector<Tensor*> parameters() {
        return {};
    }

    const std::vector<size_t>& getOutputShape() const {
        return OutputShape;
    }
};




// Convolution Layer
class Conv2dLayer : public Layer {
public:

    Tensor Kernel;
    Tensor Bias;

    size_t Stride;
    size_t Padding;

    Conv2dLayer(const std::vector<size_t>& InputShape, const std::vector<size_t>& KernelShape, 
        size_t stride, size_t padding);

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;

    std::vector<Tensor*> parameters() override {
        return { &Kernel, &Bias };
    }
};


// Activation Layer
class ActivationLayer : public Layer {
public:

    std::string Type;

    ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type);

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;
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

    PoolingLayer(const std::vector<size_t>& InputShape, size_t kh, size_t kw, size_t sh,size_t sw, 
        size_t ph, size_t pw, const std::string& type);

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;
};


// Flatten Layer
class FlattenLayer : public Layer {
public:

    FlattenLayer(const std::vector<size_t>& InputShape);

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;
};


// Dense / Linear Layer
class DenseLayer : public Layer {
public:

    Tensor Weight;
    Tensor Bias;

    size_t OutFeatures;

    DenseLayer(const std::vector<size_t>& InputShape, size_t OutNumFeature);

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;

    std::vector<Tensor*> parameters() override {
        return { &Weight, &Bias };
    }
};


// Output Layer
class OutputLayer : public Layer {
public:

    std::string Type;

    OutputLayer(const std::vector<size_t>& InputShape, const std::string& type = "CrossEntropy");

    void forward(const Tensor& input, Tensor& output, ExecutionContext& context) override;
};


// Sequential Model
class Sequential {
private:

    std::vector<std::unique_ptr<Layer>> Layers;

    std::vector<size_t> InputShape;

    size_t BatchSize = 1;
    size_t NumBatch = 0;

private:

    //const std::vector<size_t>& getCurrentOutputShape() const;

    void compileInference(ExecutionContext& Context);

    void compileTraining(ExecutionContext& Context);

    Tensor& forwardInference(ExecutionContext& Context, const Tensor& Input);

public:

    Sequential() = default;

    //----------------------------------------------------------

    void Input(const std::vector<size_t>& Shape, size_t Batch);

    //----------------------------------------------------------

    void Conv2D(const std::vector<size_t>& KernelShape, size_t Stride = 1, size_t Padding = 0);

    //----------------------------------------------------------

    void Activate(const std::string& Type);

    //----------------------------------------------------------

    void MaxPooling(size_t KH, size_t KW, size_t SH = 1, size_t SW = 1, size_t PH = 0, size_t PW = 0);

    //----------------------------------------------------------

    void AvgPooling(size_t KH, size_t KW, size_t SH = 1, size_t SW = 1, size_t PH = 0, size_t PW = 0);

    //----------------------------------------------------------

    void Flatten();

    //----------------------------------------------------------

    void Dense(size_t OutFeatures);

    //----------------------------------------------------------

    void Output(const std::string& Loss = "CrossEntropy");

    //----------------------------------------------------------

    Tensor& Predict(ExecutionContext& Context, const Tensor& Input);

    //----------------------------------------------------------

    void Train(ExecutionContext& Context, const Tensor& Input, const Tensor& Label);

    //----------------------------------------------------------

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> Params;

        for (auto& Layer : Layers) {
            auto P = Layer->parameters();

            Params.insert(Params.end(), P.begin(), P.end());
        }

        return Params;
    }
};
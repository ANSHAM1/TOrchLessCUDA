#ifndef __CNN_ARCH__
#define __CNN_ARCH__

#include "Tensor.cuh"

template<typename T>
concept FloatingTensorType = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, __half>;

_AM_START

template<FloatingTensorType T>
class Layer {
protected:
    Tensor<T> Input;
    std::vector<size_t> OutputShape;

public:
    virtual void forward(const Tensor<T>& input, Tensor<T>& output) = 0;

    virtual void backward(const Tensor<T>& upstreamGrad, Tensor<T>& downstreamGrad) = 0;

    virtual const std::vector<size_t>& getOutputShape() const {
        return OutputShape;
    }

    bool SkipInInference = false;
    virtual ~Layer() {}
};

template<FloatingTensorType T>
class Conv2dLayer : public Layer<T> {
public:
    Tensor<T> Kernel;
    size_t Stride, Padding;

    Conv2dLayer(const std::vector<size_t>& InputShape, size_t Cout, size_t Cin, size_t kH, size_t kW, size_t s, size_t p)
        : Stride(s), Padding(p) {

        Kernel = Tensor<T>({ Cout, Cin, kH, kW });
        Kernel.fillRandomOptimized(1234ULL);

        size_t Hout = (InputShape[2] + 2 * Padding - kH) / Stride + 1;
        size_t Wout = (InputShape[3] + 2 * Padding - kW) / Stride + 1;
        OutputShape = { InputShape[0], Cout, Hout, Wout };
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output, bool isTraining) override {
        if (is_training) {
            Input = Tensor<T>(input.shape(), input.data());
        }
    };
};

template<FloatingTensorType T>
class ActivationLayer : public Layer<T> {
public:
    Tensor<T> Input;
    std::string Type;

    ActivationLayer(const std::vector<size_t>& InputShape, const std::string& type)
        : Type(type) {

        if (Type != "relu" && Type != "tanh" && Type != "sigmoid")
            throw std::runtime_error("Not a valid activation function");
        OutputShape = InputShape;
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

template<FloatingTensorType T>
class PoolingLayer : public Layer<T> {
public:
    Tensor<T> Input;
    size_t kH, kW, Stride, Padding;
    std::string Type;

    PoolingLayer(const std::vector<size_t>& InputShape, size_t kh, size_t kw, size_t s, size_t p, const std::string& type) 
        : kH(kh), kW(kw), Stride(s), Padding(p), Type(type) {

        if (type != "max" && type != "avg") 
            throw std::runtime_error("invalid pooling type");

        int Hout = ((InputShape[2] - kH + 2 * Padding) / Stride) + 1;
        int Wout = ((InputShape[3] - kW + 2 * Padding) / Stride) + 1;
        OutputShape = { InputShape[0], InputShape[1], Hout, Wout };
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

template<FloatingTensorType T>
class BatchNormLayer : public Layer<T> {
public:
    Tensor<T> Input, Gamma, Beta;

    BatchNormLayer(const std::vector<size_t>& InputShape) {
        Gamma = Tensor<T>({ 1, InputShape[1], 1, 1 });
        Gamma.fill((T)1);
        Beta.zeros({ 1, InputShape[1], 1, 1 });

        OutputShape = InputShape;
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

template<FloatingTensorType T>
class DropoutLayer : public Layer<T> {
public:
    Tensor<T> Input, Mask;
    const unsigned float Prob;

    DropoutLayer(const std::vector<size_t>& InputShape, const unsigned float p)
        : Prob(p) {

        Mask.zeros(InputShape);

        OutputShape = InputShape;
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

template<FloatingTensorType T>
class DenseLayer : public Layer<T> {
public:
    Tensor<T> Input, Weight, Bias;
    size_t OutNumFeature;

    DenseLayer(const std::vector<size_t>& InputShape, size_t outNumFeature)
        : OutNumFeature(outNumFeature) {

        int InNumFeature = shape_product(InputShape) / InputShape[0];

        Weight = Tensor<T>({ OutNumFeature, InNumFeature, 1, 1 });
        Bias.zeros({ 1, 1, 1, OutNumFeature });

        OutputShape = { InputShape[0], 1, 1, OutNumFeature };
	}

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

template<FloatingTensorType T>
class OutputLayer : public Layer<T> {
public:
    Tensor<T> Input;
    std::string Type;

    OutputLayer(const std::vector<size_t>& InputShape, const std::string type) 
        : Type(type) {

        if (Type != "softmax" && Type != "sigmoid" && Type != "linear")
            throw std::runtime_error("Not a valid output function");
        OutputShape = InputShape;
    }

    void forward(const Tensor<T>& Input, Tensor<T>& Output) override {};
};

    
template<FloatingTensorType T>
class Sequential {
private:
    std::vector<size_t> InputShape;
    std::vector<std::unique_ptr<Layer<T>>> Layers;

    Tensor<T> workspace_A_;
    Tensor<T> workspace_B_;
    bool is_built_ = false;

    const std::vector<size_t>& getCurrentOutputShape() {
        if (Layers.empty()) {
            if(InputShape.empty()) 
                throw std::runtime_error("Input shape not set. Call Input() before adding layers.");
            return InputShape;
        }
        return Layers.back()->getOutputShape();
    }

public:
    Sequential() = default;

    void Input(const std::vector<size_t>& inputShape) {
        if (inputShape.size() != 4) {
            throw std::invalid_argument("Input shape must be of size 4 -> (N, C, H, W)");
        }
        InputShape = inputShape;
    }

    void Conv2D(size_t Cout, size_t Cin, size_t kH, size_t kW, size_t s = 1, size_t p = 0) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<Conv2dLayer<T>>(CurrentShape, Cout, Cin, kH, kW, s, p));
    }

    void Activate(const std::string& type) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<ActivationLayer<T>>(CurrentShape, type));
    }

    void MaxPooling(size_t kh, size_t kw, size_t s = 1, size_t p = 0) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, s, p, "max"));
	}

    void AvgPooling(size_t kh, size_t kw, size_t s = 1, size_t p = 0) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<PoolingLayer<T>>(CurrentShape, kh, kw, s, p, "avg"));
    }

    void BatchNorm() {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<BatchNormLayer<T>>(CurrentShape));
    }

    void Dropout(const unsigned float p) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<DropoutLayer<T>>(CurrentShape, p));
    }

    void Dense(int outNumFeature) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<DenseLayer<T>>(CurrentShape, outNumFeature));
	}

    void Output(const std::string type) {
        const auto& CurrentShape = getCurrentOutputShape();
        Layers.push_back(std::make_unique<OutputLayer<T>>(CurrentShape, type));
	}

    void Compile() {
        if (InputShape.empty()) {
            throw std::runtime_error("Input shape not set. Call Input() before build().");
        }
        if (Layers.empty()) {
            return;
        }

        size_t max_elements_A = 0;
        size_t max_elements_B = 0;

        max_elements_A = shape_product(InputShape);

        std::vector<size_t> current_shape = InputShape;

        for (size_t i = 0; i < Layers.size(); ++i) {
            const auto& OutputShape = Layers[i]->getOutputShape();
            size_t OutputElements = shape_product(OutputShape);

            if (i % 2 == 0) { // Writes to B
                if (OutputElements > max_elements_B) max_elements_B = OutputElements;
            }
            else { // Writes to A
                if (output_elements > max_elements_A) max_elements_A = output_elements;
            }
            current_shape = output_shape;
        }

        workspace_A_.resize({ max_elements_A });
        workspace_B_.resize({ max_elements_B });

        is_built_ = true;
        std::cout << "Model built successfully." << std::endl;
        std::cout << "Workspace A size: " << (max_elements_A * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Workspace B size: " << (max_elements_B * sizeof(T)) / (1024.0 * 1024.0) << " MB" << std::endl;
    }


    // --- NEW: The forward pass execution ---
    Tensor<T>& forward(const Tensor<T>& input) {
        if (!is_built_) {
            throw std::runtime_error("Model not built. Call build() before forward().");
        }
        if (input.shape() != input_shape_) {
            throw std::invalid_argument("Input tensor shape does not match model's expected input shape.");
        }

        // The initial input data is our first "source" tensor.
        const Tensor<T>* source = &input;
        Tensor<T>* destination = &workspace_B_; // First layer writes to B

        for (size_t i = 0; i < layers_.size(); ++i) {
            // Determine the correct input and output buffers for this layer.
            if (i % 2 == 0) {
                source = (i == 0) ? &input : &workspace_A_;
                destination = &workspace_B_;
            }
            else {
                source = &workspace_B_;
                destination = &workspace_A_;
            }

            // The destination tensor needs to be reshaped to what the layer expects for its output.
            // This does NOT reallocate memory, just changes the shape metadata.
            destination->resize(layers_[i]->get_output_shape());

            // Execute the layer's forward pass.
            layers_[i]->forward(*source, *destination);
        }

        // The final result will be in the last written-to buffer.
        if ((layers_.size() - 1) % 2 == 0) {
            return workspace_B_;
        }
        else {
            return workspace_A_;
        }
    }

//    shared_ptr<BatchWrapper> CONFUSION_MATRIX;
//    float ACCURACY;
//    vector<float> PRECISION, RECALL, F1;
//
//    int NUM_BATCHS;
//
//    vector<shared_ptr<BatchWrapper>> BWs;
//    vector<shared_ptr<Layer>> LAYERS;
//
//    vector<shared_ptr<BatchWrapper>> TestBWs;
//    vector<shared_ptr<Layer>> TestLAYERS;
//
//    bool isTrained = false;
//    bool isTested = false;
//
//    void Input(int b, int c, int h, int w, int miniBatchSize);
//    void Conv2D(int out_channels, int in_channels, int kernel_h, int kernel_w, int stride, int padding);
//    void Activation(const string& type);
//    void Pooling(int kHeight, int kWidth, int stride, int padding, const string& type);
//    void BatchNorm();
//    void Dropout(float p);
//    void Dense(int NumFeatures);
//
//private:
//    shared_ptr<OutputLayer> OutputprivateLayer(const vector<vector<vector<vector<float>>>>& miniBatch);
//
//public:
//    void Train(const vector<vector<vector<vector<float>>>>& inputs, const vector<vector<vector<vector<float>>>>& labels,
//        int epochs, float lr, float R2, float clipGrad);
//
//    void Test(const vector<vector<vector<vector<float>>>>& tensor, const vector<vector<vector<vector<float>>>>& labels);
//    vector<int> Predict(const vector<vector<vector<vector<float>>>>& tensor);
//    void showEvaluation();
//
//
//private:
//    vector<int> TP, TN, FP, FN;
//
//    vector<vector<vector<vector<float>>>> getMiniBatch(const vector<vector<vector<vector<float>>>>& batch, int idx) const;
//
//    void calculateTP_FP_FN_TN(vector<float> matrix, int C);
//    void Evaluate();
};


_AM_END

#endif // !__CNN_ARCH__
